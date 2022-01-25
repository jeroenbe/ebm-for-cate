# Identifiable Energy-based Representations: An Application to Estimating Heterogeneous Causal Effects </br><sub><sub>Yao Zhang*, Jeroen Berrevoets*, Mihaela van der Schaar [AISTATS, 2022 (forthcoming)]</sub></sub>

In this repository we provide code for our paper on EBMs for CATE estimation. The structure of this repository is as follows:
```bash
src/
|_ models/
  |_ benchmarks.py
  |_ cate_model.py
  |_ nce_ite.py
|_ data/
  |_ data_module.py
notebooks/
  |_ CATE tests.ipynb     # notebook to generate results related to CATE
  |_ Covar.ipnyb          # notebook to generate Figure 1 (right)
```

Learning our proposed model amounts to using the CLI provided in `src/models/nce_ite.py`, which is easily done using the following:
```bash
python -m src.models.nce_ite
```
the provided CLI has following options:
```
Options:
  --lr FLOAT                                    (learning rate)
  --epochs INTEGER                              (maximum amount of epochs)
  --batch_size INTEGER                          (batch size)
  --K INTEGER                                   (representation size)
  --b INTEGER                                   (amount of noisy samples)
  --wb_run TEXT                                 (weights and biases run to upload model/logs to)
  --data TEXT                                   (data to be used, one of: 'ihdp', 'synth', or 'twins')
  --perturbation_method TEXT                    (type of perturbation, one of: 'binomial', 'fix')
  --_noise_prob FLOAT                           (when perturbation method is 'binomial', this is the probability of being perturbed)
  --_noise_count INTEGER                        (when perturbation method is 'fix', these are the mount of perturbed columns)
  --n_layers INTEGER                            (amount of hidden leayers in the EBM)
  --layer_width INTEGER                         (width of the hidden layers)
  --data_location TEXT                          (location of the data; default values provided in data_module.py)
  --load_synth BOOLEAN                          (when a synth dataset is generated, it is also saved locally, this avoids regeneration)
  --test_cate BOOLEAN                           (if true, the EBM training will end with placeholder CATE learner -> only indicative and not used to generate experimental results in our paper)
  --train_size INTEGER                          (amount of samples to train on)
  --scale_treatment_balance FLOAT               (when data is 'synth', this can increase or decrease treatment imbalance, we do not use this to generate results in our paper)
  --fix_dim BOOLEAN                             (when data is 'synth', a true value fixes the dimension count to 100)
  --fix_contrastive_learning_seed BOOLEAN       (when true, the same noisy samples are sampled for each learning iteration, this is not used to generate results in our paper)
```

Our code relies on the following packages:
* to build models, we use
    - `scikit-learn 0.24.2`
    - `econml 0.11.0`
    - `pytorch 1.8.1`
    - `pytorch-lightning 1.3.1`
    - `scipy 1.4.1`
    - `numpy 1.20.3`
* to handle data, we use
    - `pandas 1.2.4`
* to log and present results we use
    - `wandb 0.10.30`
    - `matplotlib 3.4.2`

Note that we have not provided an end-to-end scheme to generate results directly from training. This is due to our use of weights and biases, which we use to save model weights.

*Equal contribution
