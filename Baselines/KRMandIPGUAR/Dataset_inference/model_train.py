import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_data_init import data_init, model_init
from params import args_parse


# train per round
def train_epoch(args, loader, model, teacher=None, lr_schedule=None, epoch_i=None, opt=None, stop=False):

    train_loss = 0; train_acc = 0; train_n = 0; i = 0

    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    alpha, T = 1.0, 1.0
    func = tqdm if stop == False else lambda x:x

    for batch_idx, (data, target) in enumerate(func(loader)):
        X, y = data.to(args.device), target.to(args.device)
        yp = model(X)

        if teacher is not None:
            with torch.no_grad():
                t_p = teacher(X).detach()
                y = t_p.max(1)[1]
            if args.exact_mode in ["extract-label", "fine-tune"]:
                loss = nn.CrossEntropyLoss()(yp, t_p.max(1)[1])
            else:
                loss = criterion_kl(F.log_softmax(yp / T, dim=1), F.softmax(t_p / T, dim=1)) * (alpha * T * T)
        else:
            loss = nn.CrossEntropyLoss()(yp, y)

        lr = lr_schedule(epoch_i + (i + 1) / len(loader))
        opt.param_groups[0].update(lr=lr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item() * y.size(0)
        train_acc += (yp.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        i += 1

        if stop:
            break

    return train_loss / train_n, train_acc / train_n


# for test
def test_epoch(args, loader, model):

    test_loss = 0; test_acc = 0; test_n = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            X, y = data.to(args.device), target.to(args.device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            test_loss += loss.item()*y.size(0)
            test_acc += (yp.max(1)[1] == y).sum().item()
            test_n += y.size(0)

    return test_loss / test_n, test_acc / test_n


# python model_train.py --data_name 'CIFAR10' --exact_mode 'teacher' --train_epochs 20 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR10' --exact_mode 'prune' --train_epochs 20 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR10' --exact_mode 'distillation' --train_epochs 20 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR10' --exact_mode 'extract-label' --train_epochs 10 --pseudo_labels 1
# python model_train.py --data_name 'CIFAR10' --exact_mode 'extract-logit' --train_epochs 10 --pseudo_labels 1
# python model_train.py --data_name 'CIFAR10' --exact_mode 'zero-shot' --train_epochs 5 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR10' --exact_mode 'fine-tune' --train_epochs 5 --pseudo_labels 1
# python model_train.py --data_name 'CIFAR100' --exact_mode 'teacher' --train_epochs 20 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR100' --exact_mode 'prune' --train_epochs 30 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR100' --exact_mode 'distillation' --train_epochs 30 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR100' --exact_mode 'extract-label' --train_epochs 20 --pseudo_labels 1
# python model_train.py --data_name 'CIFAR100' --exact_mode 'extract-logit' --train_epochs 20 --pseudo_labels 1
# python model_train.py --data_name 'CIFAR100' --exact_mode 'zero-shot' --train_epochs 30 --pseudo_labels 0
# python model_train.py --data_name 'CIFAR100' --exact_mode 'fine-tune' --train_epochs 10 --pseudo_labels 1
# *** main function for model training ***/
def train(args):

    # 1. load the xx
    train_loader, test_loader = data_init(args.data_name, args.batch_size, pseudo_labels=args.pseudo_labels)

    # 2. load student and teacher model
    teacher, student = model_init(args.data_name, args.exact_mode, args.num_class, args.pseudo_labels, args.device, temp='model')

    # 3. model training
    print("Model training...")
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lambda t: np.interp([t], [0, args.train_epochs * 2 // 5, args.train_epochs * 4 // 5, args.train_epochs],
                                      [args.lr_min, args.lr_max, args.lr_max / 10, args.lr_min])[0]

    for t in range(args.train_epochs):
        lr = lr_schedule(t)
        student.train()
        train_loss, train_acc = train_epoch(args, train_loader, student, teacher=teacher, lr_schedule=lr_schedule, epoch_i=t, opt=optimizer)
        student.eval()
        test_loss, test_acc = test_epoch(args, test_loader, student)
        print(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f}, lr: {lr:.5f}')

    torch.save(student.state_dict(), f"{args.model_dir}/final.pt")


if __name__ == "__main__":

    parser = args_parse()
    args = parser.parse_args()
    print(args)

    # set model saving path
    model_dir = f"./models/{args.data_name}/model_{args.exact_mode}"
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print("Model Directory:", model_dir)

    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.cuda.set_device(device)
    print(device)

    n_class = {"CIFAR10": 10, "CIFAR100": 100}
    args.num_class = n_class[args.data_name]

    torch.manual_seed(args.seed)

    train(args)


