# LoGoFair

This repository is built based on PyTorch, containing the code for the paper: '[AAAI 2025] LoGoFair: Post-Processing for Local and Global Fairness in Federated Learning'.

## Installation

The code is tested on Python 3.12. The dependencies can be installed using the following command. (Before proceeding, it is recommended to create a new conda environment.)

```bash
pip install -r requirements.txt
```

## Dataset

- Adult
  
  - Download the following files:
  
  - https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
    
    https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names
  
  - Place it in the folder: data/adult/raw_data

- ENEM
  
  - Download from https://download.inep.gov.br/microdados/microdados_enem_2020.zip
  
  - unzip microdados_enem_2020.zip, place them in the folder: data/enem/raw_data

- CelebA
  
  - Download files from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    
    - img_align_celeba.7z
    
    - identity_CelebA.txt
    
    - list_attr_celeba.txt
  
  - unzip img_align_celeba.7z, place them in the folder: data/celeba/raw_data

## Running the Code

**Run main.py to train the model.**

Example: Run fedavg and logofair with Adult.

```
python main.py --algorithm fedavg --data adult --num_round 50 --local_lr 0.001

python main.py --algorithm logofair --data adult --client 5 --alpha 0.5 --fairness_metric DP --post_round 20
```

For additional details on the parameters, please refer to the annotations in options.py.
