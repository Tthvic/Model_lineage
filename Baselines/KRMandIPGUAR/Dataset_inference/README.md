# Knowledge Representation of Training Data with Adversarial Examples Supporting Decision Boundary 

Code for Knowledge Representation of Training Data with Adversarial Examples Supporting Decision Boundary

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
**Running experiments of dataset inference.**

(1) Run the model training code and get teacher and student model (CIFAR10 and CIFAR100 dataset will be downloaded automatically):
```shell
python model_train.py --data_name $DATA_NAME --exact_mode $EXACT_MODE --train_epochs $TRAIN_EPOCHS --pseudo_labels $PSEUDO_LABELS
```
> `data_name` - Dataset used.\
> `exact_mode` - Type of suspected model, choice=['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher'].\
> `train_epochs` -  Number of model training epochs.\
> `pseudo_labels` - Whether the alternative dataset is used.

(2) Run the embedding generation code and generate embeddings with MinAD for dataset inference:
```shell
python embedding_compute.py --data_name $DATA_NAME --exact_mode $EXACT_MODE --feature_mode 'MinAD'
```
> `data_name` - Dataset used.\
> `exact_mode` - Type of suspected model, choice=['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher'].\
> `feature_mode` - Type of adversarial example generation, choice=['MinAD', 'MinAD_KRM'].

(3) Run the embedding generation code and generate embeddings with MinAD combined with KRM for dataset inference:
```shell
python knowledge_matrix.py --data_name $DATA_NAME --exact_mode $EXACT_MODE --feature_mode 'MinAD_KRM'
```
> `data_name` - Dataset used.\
> `exact_mode` - Type of suspected model, choice=['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher'].\
> `feature_mode` - Type of adversarial example generation, choice=['MinAD', 'MinAD_KRM'].

(4) Run the dataset inference code and test:
```shell
python dataset_inference.py --data_name $data_name --feature_mode $feature_mode
```
> `data_name` - Dataset used.\
> `feature_mode` - Type of adversarial example generation, choice=['MinAD', 'MinAD_KRM'].

## Contents
- **model_data_init.py:**
Initialization of dataset and model (teacher and student).

- **model_train.py:**
The function of training teacher and student model.

- **embedding_compute.py:**
The function of adversarial example generation. We introduce our MinAD and generate embeddings for dataset inference.

- **knowledge_matrix.py:**
The function of knowledge representation generation. We introduce KRM and generate KRM with MinAD.

- **dataset_inference.py:**
The function of dataset inference, including the process and test of dataset inference.

- **params.py:**
Introduction of parameters we used.

- **wideresnet.py:**
Network building of wide resnet.
