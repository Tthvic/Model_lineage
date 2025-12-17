# Knowledge Representation of Training Data with Adversarial Examples Supporting Decision Boundary 

Code for Knowledge Representation of Training Data with Adversarial Examples Supporting Decision Boundary
Original code repository: Knowledge Representation of Training Data with Adversarial Examples Supporting Decision Boundaries.
https://www.dropbox.com/s/upqzelbbtud8c6n/KR-Code.zip?dl=0

## Getting Started
To run this repository, we kindly advise you to install python 3.7 and PyTorch 1.8.1 with Anaconda. You may download Anaconda and read the installation instruction on the official website (https://www.anaconda.com/download/).
Create a new environment and install PyTorch and torchvision on it:

```shell
conda create --name xxx - pytorch python == 3.7
conda activate xxx - pytorch
conda install pytorch == 1.8.1
conda install torchvision -c pytorch
```

Install other requirements:
```shell
pip install numpy scikit-learn matplotlib os random copy time tqdm argparse
```

## Running experiments
**Running experiments of IPGuard.**

(1) Run the model training code and get targeted and suspected model (CIFAR10 and CIFAR100 dataset will be downloaded automatically):
```shell
python model_train.py --data_name $DATA_NAME --exact_mode $EXACT_MODE --train_epochs $TRAIN_EPOCHS
```
> `data_name` - Dataset used.\
> `exact_mode` - Type of suspected model, choice=['teacher', 'fine-tune', 'retrain', 'prune', 'SA', 'DA-LENET', 'DA-VGG'].\
> `train_epochs` -  Number of model training epochs.

(2) Run the adversarial example generation code and generate adversarial examples with MinAD for IPGuard:
```shell
python adversarial_generation.py --data_name $DATA_NAME --feature_mode 'MinAD'  
``` 
> `data_name` - Dataset used.\  
> `feature_mode` - Type of adversarial example generation, choice=['MinAD', 'MinAD_KRM'].

(3) Run the adversarial example generation code and generate adversarial examples with MinAD combined with KRM for IPGuard:
```shell
python knowledge_matrix.py --data_name $DATA_NAME --feature_mode 'MinAD_KRM'
```
> `data_name` - Dataset used.\
> `feature_mode` - Type of adversarial example generation, choice=['MinAD', 'MinAD_KRM'].

(4) Run the IPGuard code and test:
```shell
python ipguard_function.py --data_name $DATA_NAME --feature_mode $FEATURE_MODE
```
> `data_name` - Dataset used.\
> `feature_mode` - Type of adversarial example generation, choice=['MinAD', 'MinAD_KRM'].

## Contents
- **model_data_init.py:**
Initialization of dataset and model (teacher and student).

- **model_train.py:**
The function of training teacher and student model.

- **adv_generation.py:**
The function of adversarial example generation. We introduce our MinAD and generate adversarial examples for IPGuard.

- **knowledge_matrix.py:**
The function of knowledge representation generation. We introduce KRM and generate KRM with MinAD.

- **ipguard_function.py:**
The function of ipguard_function, including the process and test of ipguard_function.

- **params.py:**
Introduction of parameters we used.

- **wideresnet.py:**
Network building of wide resnet.

- **resnet.py:**
Network building of resnet.
