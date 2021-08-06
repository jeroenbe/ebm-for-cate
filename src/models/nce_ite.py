import random, math, click, wandb

from collections import defaultdict

from econml import dr

import numpy as np
from scipy.stats import norm

import torch
import torch.nn as nn

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data.data_module import IHDP, LBIDD, Twins, Synth
from src.models.cate_model import CATEModel



class NCE(pl.LightningModule):
    def __init__(
            self,
            input_dim,
            K, b,
            lr=.001,
            _noise_count: int=5,
            _noise_prob: float=.5,
            perturbation_method: str='fix',
            architecture=((20, 20), (20, 20)),
            cate_learner_no_repr=None,
            cate_learner_repr=None,
            fix_contrastive_learning_seed: bool=False):

        super().__init__()

        self.K = K
        self.input_dim = input_dim
        self.lr = lr

        self.architecture = architecture
        self.fix_contrastive_learning_seed = fix_contrastive_learning_seed

        layers = [nn.Sequential(nn.ReLU(), nn.Linear(layer[0], layer[1])) for layer in self.architecture]

        self.f = nn.Sequential(
            nn.Linear(self.input_dim, self.architecture[0][0]),
            *layers,
            nn.Linear(self.architecture[-1][1], self.K)
        )

        
        if fix_contrastive_learning_seed:
            pl.utilities.seed.seed_everything(0)

        self.B = self.prep_matrix()
        self.b = b
        self.b_factorial = math.factorial(b)

        self._noise_count = _noise_count                # Amount of features to perturb
        self._noise_prob = _noise_prob                  # When perturbation_method is binomial
        self.perturbation_method = perturbation_method  # How perturbation happens

        self.cate_learner_repr = cate_learner_repr
        self.cate_learner_no_repr = cate_learner_no_repr
        
        self.save_hyperparameters()

    # NCE METHODS
    # -----------
    def prep_matrix(self):
        # random gaussian
        np.random.seed(1)
        B = np.random.normal(size=(self.K, self.K))
        # orthogonalise (eigen decom)
        _, B = np.linalg.eigh(B)
        return B
    
    def _sigma(self, X, j: int, p_j: float):
        beta = self.B[:,j]          # get jth column of B
        beta = torch.tensor(beta, device=self.device)
        out = beta.T @ self.f(X)    # get product of beta and representation
        
        return out - torch.log(p_j)

    def _get_columns_to_corrupt(self, method: str='fix'):
        d = self.trainer.datamodule
        if method == 'fix':
            assert self._noise_count is not None
            m = random.sample(range(len(d.x_cols)), self._noise_count)                  # Get the features to perturb
        elif method == 'binomial':
            assert self._noise_prob is not None
            mask = np.random.binomial(p=self._noise_prob, n=1, size=len(d.x_cols))
            m = np.array(list(range(len(d.x_cols))))[mask == 1]

        return m


    def get_contrastive_samples(self, X, j, perturbation_method: str='fix'):

        d = self.trainer.datamodule

        m = self._get_columns_to_corrupt(method=perturbation_method)

        
        p_tilde_sample = d.train[d.train.k == j].sample(self.b, replace=True)       # Sample b rows from D
        p_tilde_sample = p_tilde_sample[d.x_cols]

        
        p_tilde_j = torch.ones(self.b)
        for m_j in m:
            col = d.x_cols[m_j]
            possible_noise = d.unique_values[col]
            true_values = p_tilde_sample[col].to_numpy()
            if possible_noise is not None and len(possible_noise)>1:                                          # We randomly select self.noise_count
                p_tilde_j *= 1 / (len(possible_noise))                              # dimensions to perturb. The perturbation
                #perturbations = [                                                   # is simply selecting a random value from
                #    np.random.choice(                                               # the previously seen row values. Each sample
                #        tuple(set(possible_noise) - {true_value})                   # then has a uniform probability of being
                #    ) for true_value in true_values]                                # selected.
                
                p_tilde_sample[col] = np.random.choice(possible_noise, size=self.b) # TEMP: include true_value as option
                                                                                    
            else:
                eps = np.random.normal(size=self.b)                                 # When the column is not categorical
                p_tilde_sample[col] += eps                                          # we perturb with white noise. p_tilde_j
                p_tilde_j *= norm.pdf(eps)                                          # is then obtained by the pdf of N(0;1)
        p_tilde_sample = torch.tensor(p_tilde_sample.to_numpy(), device=self.device)

        p_tilde_j *= self.p_m if perturbation_method == 'fix' else 1/(self._noise_prob ** len(m))

        return p_tilde_sample, p_tilde_j                                            # p_m is 1 / dCb


    def _elementwise_loss(self, X, j, log: str=None, perturbation_method: str='fix', seed: int=0):
        if seed > 0:
            np.random.seed(seed * self.current_epoch)


        X_, p_j = self.get_contrastive_samples(
            X, j, perturbation_method=perturbation_method)              # yields b x |X| matrix
        sigma_x = self._sigma(X, j=j, p_j=torch.tensor(1.))             # log(1) = 0 => the log-term is omitted

        sum = 0
        j_ = self.cluster.predict(X_.cpu().data)
        for i, (X_i) in enumerate(X_):
            sigma_x_i = self._sigma(X_i, j=j_[i], p_j=p_j[i])
            sum += torch.exp(sigma_x_i)

        negative_log = -torch.log(torch.exp(sigma_x) + sum)

        if log:
            self.log(f'{log} | negative_log', negative_log, on_epoch=True)
            self.log(f'{log} | sum', sum, on_epoch=True)
            self.log(f'{log} | sigma_x', sigma_x, on_epoch=True)
            self.log(f'{log} | sigma_x + negative_log', sigma_x.item() + negative_log.item(), on_epoch=True)

        return - (sigma_x + negative_log)


    # TORCH METHODS
    # -----------
    def forward(self, x):
        return self.f(x)

    # LIT METHODS
    # -----------
    def on_train_end(self, stage=None):
        # INFO: We should report val-performance for a CATE-learner
        if self.cate_learner_no_repr is not None and self.cate_learner_repr is not None:
            cate_nr = CATEModel(model=self.cate_learner_no_repr, representation=None)
            cate_r = CATEModel(model=self.cate_learner_repr, representation=self)

            dm = self.trainer.datamodule

            X_train = dm.train[dm.x_cols]
            Z_train = dm.train.z
            Y_train = dm.train.y

            X_test = dm.test[dm.x_cols]
            Y0_test= dm.test.y0
            Y1_test= dm.test.y1

            cate_nr.fit(X_train, Z_train, Y_train)
            cate_r.fit(X_train, Z_train, Y_train)

            pehe_no_repr = cate_nr.eval(X_test, Y0_test, Y1_test)
            pehe_repr = cate_r.eval(X_test, Y0_test, Y1_test)

            self.logger.log_metrics({
                'PEHE with representation': pehe_repr,
                'PEHE without representation': pehe_no_repr})


    def on_fit_start(self, stage=None):
        self.cluster = KMeans(n_clusters=self.K)

        d = len(self.trainer.datamodule.x_cols)
        dCb = math.factorial(d) // math.factorial(self.b) // math.factorial(d - self.b)
        self.p_m = 1 / dCb

        X, _, _ = self.trainer.datamodule.train_dataloader().dataset.dataset.tensors

        self.cluster.fit(X)

        self.trainer.datamodule.train['k'] = self.cluster.predict(X)
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['cluster'] = self.cluster
        checkpoint['B'] = self.B
    
    def on_load_checkpoint(self, checkpoint):
        self.cluster = checkpoint['cluster']
        self.B = checkpoint['B']

    def configure_optimizers(self):
        return torch.optim.Adam(self.f.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, log='Train')
        self.log('training loss', -loss.item(), on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', -loss.item(), on_epoch=True)
        
        return loss

    def shared_step(self, batch, log: str=None):
        X, _, _ = batch
        j = self.cluster.predict(X.cpu().data)
        loss = 0
        for i, X_i in enumerate(X):
            seed = i if self.fix_contrastive_learning_seed else 0
            loss += self._elementwise_loss(X_i, j=j[i], log=log, perturbation_method=self.perturbation_method, seed=seed)
        loss *= 1/X.size(0)

        return loss

@click.command()
@click.option('--lr', type=float, default=.001)
@click.option('--epochs', type=int, default=100)
@click.option('--batch_size', type=int, default=256)
@click.option('--K', type=int, default=5)
@click.option('--b', type=int, default=5)
@click.option('--wb_run', type=str, default='nce-ite')
@click.option('--data', type=str, default='ihdp')
@click.option('--perturbation_method', type=str, default='fix')
@click.option('--_noise_prob', type=float, default=.5)
@click.option('--_noise_count', type=int, default=5)
@click.option('--n_layers', type=int, default=2)
@click.option('--layer_width', type=int, default=20)
@click.option('--data_location', type=str, default='./data')
@click.option('--load_synth', type=bool, default=False)
@click.option('--test_cate', type=bool, default=False)
@click.option('--train_size', type=int, default=100)
@click.option('--scale_treatment_balance', type=float, default=1.)
@click.option('--fix_dim', type=bool, default=False)
@click.option('--fix_contrastive_learning_seed', type=bool, default=False)
def main(
        lr, 
        epochs, 
        batch_size, 
        k, 
        b, 
        wb_run, 
        data, 
        perturbation_method, 
        _noise_prob, 
        _noise_count, 
        n_layers, 
        layer_width,
        data_location,
        load_synth,
        test_cate,
        train_size,
        scale_treatment_balance,
        fix_dim,
        fix_contrastive_learning_seed,):

    torch.set_default_dtype(torch.float64)
    wb_run=f'{wb_run}-{data}'

    # LOAD DATA
    data_seed_standard = 524
    data_seeds = defaultdict(lambda: data_seed_standard)
    data_seeds[100] = 979
    data_seeds[500] = 255

    data_dims_standard = 100
    data_dims = defaultdict(lambda: data_dims_standard)
    data_dims[100] = 50
    data_dims[250] = 100
    data_dims[500] = 150
    data_dims[1000] = 200
    data_dims[1500] = 250

    dims = {
        'ihdp': 25,
        'lbidd': 168,
        'twins': 48,
        'synth': 100 if fix_dim else data_dims[train_size]
    }

    # CONSTRUCT MODEL
    if test_cate:
        kwargs = {
            #'max_depth': 40,
            #'n_estimators': 100,
            'max_iter': 8000,
            'tol': .001
        }

        cate_r = dr.SparseLinearDRLearner(
            model_propensity=SVC(probability=True),
            model_regression=KernelRidge(),
            **kwargs)
        cate_nr = dr.SparseLinearDRLearner(
            model_propensity=SVC(probability=True),
            model_regression=KernelRidge(),
            **kwargs)
    else:
        cate_r, cate_nr = None, None

    input_dim = dims[data]
    architecture = tuple([(layer_width, layer_width) for _ in range(n_layers)])
    model = NCE(
        input_dim=input_dim, 
        K=k,
        b=b,
        lr=lr,
        _noise_count=_noise_count,
        _noise_prob=_noise_prob,
        perturbation_method=perturbation_method,
        architecture=architecture,
        cate_learner_repr=cate_r,
        cate_learner_no_repr=cate_nr,
        fix_contrastive_learning_seed=fix_contrastive_learning_seed
    )

    


    

    if data == 'ihdp':
        dm = IHDP(batch_size=batch_size, limit_train_size=train_size)
    elif data == 'lbidd':
        dm =  LBIDD(
            batch_size=batch_size, id='93aab00aeb234a3b985eeb32e04a353d',
            location='./data/LBIDD')
    elif data == 'twins':
        dm = Twins(batch_size=batch_size, limit_train_size=train_size)
    elif data == 'synth':
        if fix_dim:
            dim = 100
        else:
            dim = data_dims[train_size]
        dm = Synth(
            batch_size=batch_size,
            location=data_location,
            use_existing_data=load_synth,
            n=train_size,
            seed=data_seeds[train_size],
            scale_treatment_balance=scale_treatment_balance,
            dim=dim)
    else:
        raise ValueError('Error value on --data. Please give one of "ihdp", "lbidd", or "twins".')
    
    dm.prepare_data()
    dm.setup()

    # SETUP LOGGING CALLBACKS
    wb_logger = WandbLogger(project=wb_run, log_model=True)
    wb_logger.experiment.config.train_size = train_size
    wb_logger.experiment.config.data_dim = dm.size(1)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', filename='nce.ckpt', dirpath=wb_logger.experiment.dir)
    
    # SETUP GPU
    gpus = 1 if torch.cuda.is_available() else 0

    # TRAIN NETWORK
    trainer = Trainer(logger=wb_logger, check_val_every_n_epoch=7, callbacks=[
        checkpoint_callback,
        EarlyStopping(monitor='val_loss',  patience=7),], max_epochs=epochs, gpus=gpus)
    trainer.fit(model, datamodule=dm)

    wandb.finish()








if __name__ == '__main__':
    main()
