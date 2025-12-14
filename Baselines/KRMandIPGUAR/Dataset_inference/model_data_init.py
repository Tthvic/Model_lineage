import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from wideresnet import WideResNet


def model_init(data_name, exact_mode, num_classes, pseudo_labels, device, temp):

    Net = WideResNet
    w_f = 2 if data_name == "CIFAR100" else 1
    deep_full = 28; deep_half = 16

    teacher = None; student = None

    # set the model for training
    if temp == "model":
        ## set the Teacher/Student model for different attack modes
        # 1. teacher model is loaded if attacked
        if exact_mode in ["teacher", "prune"]:
            teacher = None
        else:
            teacher = Net(n_classes=num_classes, depth=deep_full, widen_factor=10, dropRate=0.3)
            teacher = nn.DataParallel(teacher).to(device)
            path = f"./models/{data_name}/model_teacher/final"
            teacher = load(teacher, path)
            teacher.eval()

        # 2. load student model
        if exact_mode == 'zero-shot':
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f, normalize=False)
            student = nn.DataParallel(student).to(device)
            student.eval()

        elif exact_mode == "prune":
            # WRN26->WRN18
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f, dropRate=0.3)
            student = nn.DataParallel(student).to(device)
            student.train()

        elif exact_mode == "fine-tune":
            student = Net(n_classes=num_classes, depth=deep_full, widen_factor=10)
            student = nn.DataParallel(student).to(device)
            path = f"./models/{data_name}/model_teacher/final"
            student = load(student, path)
            student.train()
            assert pseudo_labels

        elif exact_mode in ["extract-label", "extract-logit"]:
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f)
            student = nn.DataParallel(student).to(device)
            student.train()
            assert pseudo_labels

        elif exact_mode == "distillation":
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f)
            student = nn.DataParallel(student).to(device)
            student.train()

        else:
            # exact_mode = ['teacher']
            student = Net(n_classes=num_classes, depth=deep_full, widen_factor=10, dropRate=0.3)
            student = nn.DataParallel(student).to(device)
            student.train()


    # set the model for embedding generation
    if temp == "feature":
        if exact_mode == 'zero-shot':
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f, normalize=False)

        elif exact_mode == "prune":
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f, dropRate=0.3)

        elif exact_mode == "fine-tune":
            student = Net(n_classes=num_classes, depth=deep_full, widen_factor=10)

        elif exact_mode in ["extract-label", "extract-logit", "distillation"]:
            student = Net(n_classes=num_classes, depth=deep_half, widen_factor=w_f)

        else:
            student = Net(n_classes=num_classes, depth=deep_full, widen_factor=10, dropRate=0.3)

    return teacher, student


def data_init(data_name, batch_size, pseudo_labels, normalize=False, train_shuffle=True):

    data_source = datasets.CIFAR10 if data_name == "CIFAR10" else datasets.CIFAR100

    tr_normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) \
        if normalize else transforms.Lambda(lambda x: x)

    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(), tr_normalize,
                                          transforms.Lambda(lambda x: x.float())])

    transform_test = transforms.Compose([transforms.ToTensor(), tr_normalize, transforms.Lambda(lambda x: x.float())])

    if not train_shuffle:
        print("No Transform")
        transform_train = transform_test

    trainset = data_source(root=f'./xx/{data_name}', train=True, download=True, transform=transform_train)
    testset = data_source(root=f'./xx/{data_name}', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=train_shuffle)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # for model extraction
    if pseudo_labels:
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])

        import pickle, os
        aux_data_filename = "ti_500K_pseudo_labeled.pickle"
        aux_path = os.path.join("./data", aux_data_filename)
        print("Loading xx from %s" % aux_path)
        with open(aux_path, 'rb') as f:
            aux = pickle.load(f)
        aux_data = aux['xx']
        aux_targets = aux['extrapolated_targets']

        cifar_train = PseudoDataset(aux_data, aux_targets, transform=transform_train)
        train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=train_shuffle)

    return train_loader, test_loader


def load(model, model_name):
    try:
        model.load_state_dict(torch.load(f"{model_name}.pt"))
    except:
        dictionary = torch.load(f"{model_name}.pt")['state_dict']
        new_dict = {}
        for key in dictionary.keys():
            new_key = key[7:]
            if new_key.split(".")[0] == "sub_block1":
                continue
            new_dict[new_key] = dictionary[key]
        model.load_state_dict(new_dict)
    return model


class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x_data = x
        self.y_data = torch.from_numpy(y).long()
        self.transform = transform
        self.len = self.x_data.shape[0]

    def __getitem__(self, index):
        x_data_index = self.x_data[index]
        if self.transform:
            x_data_index = self.transform(x_data_index)
        return x_data_index, self.y_data[index]

    def __len__(self):
        return self.len

