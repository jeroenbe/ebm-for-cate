import click, wandb, torch

import numpy as np
import torch.nn as nn

#from src.models.nce_ite import NCE
from src.data.data_module import IHDP, LBIDD, Twins, Synth

#from CATENets import SNet

from econml import dr
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class CATEModel():
    def __init__(self, model, representation=None, standardize=False,):
        self.model = model
        self.representation = representation
        self.standardize = standardize

    def _standardize(self, X):
        if self.standardize:
            if isinstance(X, torch.Tensor):
                with torch.no_grad():
                    mean = X.mean(dim=0).cpu().numpy()
                    std = X.std(dim=0).cpu().numpy()

                    #mean = torch.min(X, dim=0).values.cpu().numpy()
                    #std = torch.max(X, dim=0).values.cpu().numpy()
            else:
                mean = X.mean(axis=0)
                std = X.std(axis=0)

                #mean = X.max(axis=0)
                #std = X.min(axis=0)
        else:
            mean = 0
            std = 1


        X -= mean
        X /= std

        return X, mean, std

    def _representation(self, X):
        if self.representation is None:
            return X.values
        if not isinstance(self.representation, nn.Module):
            return self.representation(X)

        X = torch.tensor(X.values, requires_grad=False, device=self.representation.device)
        with torch.no_grad():
            X = self.representation(X)

        return X.cpu().data
    
    def fit(self, X, Z, Y, **kwargs):
        X = self._representation(X)
        X, self.mean, self.std = self._standardize(X)

        self.model.fit(Y, Z, X=X, **kwargs)

    def predict(self, X):
        X = self._representation(X)
        X -= self.mean
        X /= self.std
        return self.model.effect(X)

    def eval(self, X, Y0, Y1):
        tau = Y1.values - Y0.values
        tau_hat = self.model.effect(self._representation(X))

        return np.sqrt(((tau - tau_hat)**2).mean())

# UNCOMMENT IF A CATE MODEL IS TO BE LEARNED FROM CLI
#
# @click.command()
# @click.option('--data', type=str, default='ihdp')
# @click.option('--data_location', type=str, default='./data')
# @click.option('--repr_model', type=str, default='btuuv92k')
# @click.option('--wandb_user', type=str, default='jeroenbe')
# @click.option('--ckpt', type=str, default='nce.ckpt.ckpt')
# def main(
#         data,
#         data_location,
#         repr_model,
#         wandb_user,
#         ckpt,):

#     torch.set_default_dtype(torch.float64)
#     # LOAD DATA
#     if data == 'ihdp':
#         dm = IHDP(batch_size=256,)
#     elif data == 'lbidd':
#         dm =  LBIDD(
#             batch_size=256,
#             id='93aab00aeb234a3b985eeb32e04a353d',
#             location='../data/LBIDD')
#     elif data == 'twins':
#         dm = Twins(batch_size=256,)
#     elif data == 'synth':
#         # note that synth data HAS to exist, otherwise we can't have a representation
#         dm = Synth(batch_size=256, location=data_location, use_existing_data=True,)

#     else:
#         raise ValueError('Error value on --data. Please give one of "ihdp", "lbidd", or "twins".')
    
#     dm.prepare_data()

#     X_train = dm.train[dm.x_cols]
#     Z_train = dm.train.z
#     Y_train = dm.train.y

#     X_test = dm.test[dm.x_cols]
#     Y0_test= dm.test.y0
#     Y1_test= dm.test.y1

#     # SET SAMPLE SIZES
#     sample_sizes = np.linspace(100, len(X_train) - len(X_train)%1000, 15, dtype=int)

#     # LOAD REPRESENTATION
#     file = wandb.restore(ckpt, run_path=f'{wandb_user}/nce-ite-{data}/{repr_model}', replace=True)
#     representation = NCE.load_from_checkpoint(file.name)

#     # SETUP CATE MODELS
#     kwargs = {
#         #'max_depth': 40,
#         #'n_estimators': 100,
#         'max_iter': 20000,
#     }

#     dr_est_repr = dr.SparseLinearDRLearner(
#         model_propensity=GradientBoostingClassifier(),
#         model_regression=GradientBoostingRegressor(),
#         **kwargs)
#     dr_est_no_repr = dr.SparseLinearDRLearner(
#         model_propensity=GradientBoostingClassifier(),
#         model_regression=GradientBoostingRegressor(),
#         **kwargs)

#     dr_learner_repr = CATEModel(model=dr_est_repr, representation=representation)
#     dr_learner_no_repr = CATEModel(model=dr_est_no_repr, representation=None)
    
#     # FIT CATE MODELS
#     for size in sample_sizes:
#         dr_learner_repr.fit(X_train.iloc[:size,:], Z_train.iloc[:size], Y_train.iloc[:size])
#         dr_learner_no_repr.fit(X_train.iloc[:size,:], Z_train.iloc[:size], Y_train.iloc[:size])

#         # TEST CATE MODELS
#         # PEHE
#         pehe_repr = dr_learner_repr.eval(X_test, Y0_test, Y1_test)
#         pehe_no_repr = dr_learner_no_repr.eval(X_test, Y0_test, Y1_test)
        
#         print(f'With Representation (size={size}):', pehe_repr)
#         print(f'Without Representation:(size={size})', pehe_no_repr)



# if __name__ == '__main__':
#     main()