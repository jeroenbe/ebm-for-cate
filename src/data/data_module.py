import os

import numpy as np

import torch, joblib
import torch.nn as nn
from torch.utils.data import random_split

from sklearn.preprocessing import StandardScaler

import pytorch_lightning as pl
import pandas as pd

class CATE_DATA(pl.LightningDataModule):
    def __init__(self, location: str, train_frac: float, batch_size: int, limit_train_size: int=-1):
        super().__init__()

        assert train_frac > 0 and train_frac < 1
        self.train_frac = train_frac
        self.location = location
        self.batch_size = batch_size
        self.limit_train_size = limit_train_size

        self.unique_values = {}

    def setup(self, stage=None):
        if self.limit_train_size > 0:
            self.train = self.train.sample(self.limit_train_size)

        if stage in (None, 'fit'):
            X_tensor = torch.tensor(self.train[self.x_cols].to_numpy())
            z_tensor = torch.tensor(self.train.z.to_numpy())
            y_tensor = torch.tensor(self.train.y.to_numpy())

            for column in self.x_cols:
                uniques = self.train[column].unique()
                if len(uniques) <= 50:                      # this is an arbitrary boundary to 
                    self.unique_values[column] = uniques    # determine weither col is categorical
                else:
                    self.unique_values[column] = None       # when col is continuous
                 

            val_length = int(np.floor(X_tensor.size(0) * .2))

            D = torch.utils.data.TensorDataset(X_tensor, y_tensor, z_tensor)
            self.train_ds, self.val_ds = random_split(D, [X_tensor.size(0) - val_length, val_length])

            self.dims = (0, len(self.x_cols))
        
        if stage in (None, 'test'):
            X_tensor = torch.tensor(self.test[self.x_cols].to_numpy())
            y0_tensor = torch.tensor(self.test.y0.to_numpy())
            y1_tensor = torch.tensor(self.test.y1.to_numpy())
            
            self.test_ds = torch.utils.data.TensorDataset(X_tensor, y0_tensor, y1_tensor)
    
    def train_dataloader(self):
        bs = self.batch_size if self.batch_size < len(self.train_ds) else len(self.train_ds) -1
        return torch.utils.data.DataLoader(self.train_ds, batch_size=bs)
    def test_dataloader(self):
        bs = self.batch_size if self.batch_size < len(self.test_ds) else len(self.test_ds) -1
        return torch.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size)
    def val_dataloader(self):
        bs = self.batch_size if self.batch_size < len(self.val_ds) else len(self.val_ds) -1
        return torch.utils.data.DataLoader(self.val_ds, batch_size=self.batch_size)


class Synth(CATE_DATA):
    def __init__(
            self,
            batch_size: int,
            train_frac: float=.7,
            location: str='./data/synth',
            use_existing_data: bool=False,
            latent_dim: int=5,
            dim: int=100,
            n: int=10000,
            limit_train_size: int=-1,
            treatment_factor_size: int=-1,
            seed: int=None,
            scale_treatment_balance: float=1.):
        super().__init__(location, train_frac, batch_size, limit_train_size)

        treatment_factor_size = dim if treatment_factor_size < 0 else treatment_factor_size

        if not use_existing_data and n < 0:
            raise ValueError('If use_existing_data is False, n should be larger than 0')
        
        self.seed = seed
        self._random_seed = int(np.random.rand(1)*1000)

        pl.seed_everything(self.seed)

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.uniform_(m.weight, a=-.5, b=.5)

        self.x = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, dim),
        )

        self.y0 = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 1),
        )
        self.y1 = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 1),
        )

        self.z = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Linear(32, 32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.x.apply(init_weights)
        self.y0.apply(init_weights)
        self.y1.apply(init_weights)
        self.z.apply(init_weights)

        pl.seed_everything(self._random_seed)

        
        self.treatment_factor_size = treatment_factor_size
        self.scale_treatment_balance = scale_treatment_balance
        self.latent_dim = latent_dim
        self.use_existing_data = use_existing_data
        self.dim = dim
        self.n = n
        
    
    def prepare_data(self):
        if self.location is not None and self.use_existing_data:
            self.train = pd.read_csv(f'{self.location}/train.csv')
            self.test = pd.read_csv(f'{self.location}/test.csv')
            self.x_cols = self.train.columns.values[~np.isin(self.train.columns.values, ['z', 'y', 'y0', 'y1'])]
        
        else:
            pl.seed_everything(self.seed)

            u = np.random.normal(size=(self.n+20000, self.latent_dim), scale=1)
            self.u = u
            with torch.no_grad():
                X = self.x(torch.tensor(u)).numpy() 
                X -= X.mean(axis=0)
                X /= X.std(axis=0)
                X += np.random.normal(size=X.shape, scale=.5)
                
                Y0 = np.exp(self.y0(torch.tensor(u)).numpy())
                Y0 -= Y0.mean()
                Y0 /= Y0.std()
                Y0 += np.random.normal(size=Y0.shape)
                
                Y1 = np.exp(self.y1(torch.tensor(u)).numpy())
                Y1 -= Y1.mean()
                Y1 /= Y1.std()
                Y1 += np.random.normal(size=Y1.shape)

            self.x_cols = np.array([f'x{i}' for i in range(1, self.dim+1)])
            D = np.c_[X, Y0, Y1]
            D = pd.DataFrame(data=D, columns=[*self.x_cols, 'y0', 'y1'])

            mask = np.random.rand(len(D)) < self.train_frac

            self.train = D[:self.n].copy(deep=True)
            self.train_u = u[:self.n]
            self.test = D[self.n:].copy(deep=True)
            self.test_u = u[self.n:]
            
            with torch.no_grad():
                Z = self.z(torch.tensor(self.train_u))
                Z += np.random.normal(size=Z.shape)
                Z = torch.sigmoid(Z).numpy()
                # self.z(torch.tensor(
                #     self.train.loc[:,self.x_cols[:self.treatment_factor_size]].values)
                # ).numpy()
            
            Z += 0.001
            Z /= 1.002
            Z *= self.scale_treatment_balance       # if the data should be more imbalanced, just scale down Z

            Z = np.random.binomial(p=Z, n=1)


            self.train.loc[:,'z'] = Z

            def select_outcome(row):
                return row.y0 if row.z == 0 else row.y1

            self.train.loc[:,'y'] = self.train.apply(lambda row: select_outcome(row), axis=1)

            if not os.path.exists(self.location):
                os.makedirs(self.location)
            
            self.train.to_csv(f'{self.location}/train.csv', index=False)
            self.test.to_csv(f'{self.location}/test.csv', index=False)

            pl.seed_everything(self._random_seed)
        


class LBIDD(CATE_DATA):
    def __init__(self, id: str, batch_size: int, train_frac: float=.7, location: str='./data/LBIDD', limit_train_size: int=-1):
        super().__init__(location, train_frac, batch_size, limit_train_size)
        self.id = id
    
    def prepare_data(self):
        X = pd.read_csv(f'{self.location}/x.csv')
        self.x_cols = X.columns.values
        self.x_cols = self.x_cols[self.x_cols != 'sample_id']

        fact = pd.read_csv(f'{self.location}/scaling/factuals/{self.id}.csv')
        counterfact = pd.read_csv(f'{self.location}/scaling/counterfactuals/{self.id}_cf.csv')
    
        train_sample_id = X.sample(frac=self.train_frac)
        test_sample_id = X[~X.sample_id.isin(train_sample_id.sample_id)]
        
        self.test = pd.merge(test_sample_id, counterfact, on='sample_id')
        self.train = pd.merge(train_sample_id, fact, on='sample_id')

        self.test = self.test.drop('sample_id', axis=1)
        self.train = self.train.drop('sample_id', axis=1)
        
    

class IHDP(CATE_DATA):
    def __init__(self, batch_size: int, train_frac: float=.7, location: str='https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv', limit_train_size: int=-1):
        super().__init__(location, train_frac, batch_size, limit_train_size)

    def prepare_data(self):
        D = pd.read_csv(self.location, header=None)
        col = ['z', 'y_fact', 'y_cfact', 'mu0', 'mu1', ]
        self.x_cols = []

        for i in range(1, 26):
            col.append(f'x{i}')
            self.x_cols.append(f'x{i}')
        
        D.columns = col

        mask = np.random.rand(len(D)) < self.train_frac

        self.train = D[mask].copy(deep=True)
        self.test = D[~mask].copy(deep=True)


        self.train.rename({'y_fact': 'y'}, axis=1, inplace=True)
        self.train.drop('y_cfact', axis=1, inplace=True)

        def y_switch(row, z):
            return row.y_fact if row.z == z else row.y_cfact
        
        self.test['y0'] = self.test.apply(lambda row: y_switch(row, 0), axis=1)
        self.test['y1'] = self.test.apply(lambda row: y_switch(row, 1), axis=1)



class Twins(CATE_DATA):
    def __init__(self, batch_size: int, train_frac: float=.7, 
            location: str='https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS',
            seperate_files: dict={
                'X': 'twin_pairs_X_3years_samesex.csv',
                'Y': 'twin_pairs_Y_3years_samesex.csv',
                'Z': 'twin_pairs_T_3years_samesex.csv',
            }, limit_train_size: int=1000):
        super().__init__(location, train_frac, batch_size, limit_train_size)

        self.seperate_files = seperate_files

    def prepare_data(self):
        #_0 denotes features specific to the lighter twin and _1 denotes features specific to the heavier twin
        self.x_cols = ['pldel', 'birattnd', 'brstate', 'stoccfipb', 'mager8',
            'ormoth', 'mrace', 'meduc6', 'dmar', 'mplbir', 'mpre5', 'adequacy',
            'orfath', 'frace', 'birmon', 'gestat10', 'csex', 'anemia', 'cardiac',
            'lung', 'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper',
            'eclamp', 'incervix', 'pre4000', 'preterm', 'renal', 'rh', 'uterine',
            'othermr', 'tobacco', 'alcohol', 'cigar6', 'drink5', 'crace',
            'data_year', 'nprevistq', 'dfageq', 'feduc6', 'dlivord_min', 'dtotord_min', 
            'brstate_reg', 'stoccfipb_reg', 'mplbir_reg']

        X = pd.read_csv(f'{self.location}/{self.seperate_files["X"]}')[self.x_cols]
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=self.x_cols)

        Y = pd.read_csv(f'{self.location}/{self.seperate_files["Y"]}', index_col=0)
        
        D = X.merge(Y, left_index=True, right_index=True).dropna()

        D.rename(columns={
            'mort_0': 'y0',
            'mort_1': 'y1',
        }, inplace=True)

        mask = np.random.rand(len(D)) < self.train_frac

        self.train = D[mask].copy(deep=True)
        self.test = D[~mask].copy(deep=True)

        Z = np.random.binomial(p=.5, n=1, size=len(self.train))
        self.train.loc[:,'z'] = Z

        def select_outcome(row):
            return row.y0 if row.z == 0 else row.y1

        self.train.loc[:,'y'] = self.train.apply(lambda row: select_outcome(row), axis=1)








        



        



        






        
        


