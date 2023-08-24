# ChromEDA
## ChromEDA: Chromosome Classification based on Ensemble Domain Adaptation


This repository provides a PyTorch implementation of our work -> [[**PDF**]]() [[**arXiv**]]()
In this work, we develops a multivariate learning strategy including soft pseudo-label learning, adversarial learning, and angle classification learning to close the feature distribution between the source and target domains. 
The results of cross-domain experiments designed on public and private datasets show that the ChromEDA model can effectively improve the domain difference problem that exists in cross-domain learning and outperforms existing 
cross-domain classification methods in different cross-domain applications.

## Usage
### 1. Cloning the repository
```bash
$ git clone https://github.com/labiip/ChromEDA
$ cd ChromEDA/
```

### 2. Domain Adaptation
It is possible to train a Mix model with more than one datasets using `t_double_class_unsupervise.py`. To do that, modify `PATH` and specify the directory for the datasets in `train_dir`.
 ```bash
$ python t_double_class_unsupervise.py
```

### 3. Evaluating the model
 ```bash
$ python plot_sum_roc
```

### 4. t-sne visualization
 ```bash
$ python plot_t_sne.py
```

