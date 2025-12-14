import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model_data_init import data_init, model_init
from params import args_parse


# for training
def train_epoch(args, loader, model, lr_schedule=None, epoch_i=None, opt=None):

    train_loss = 0; train_acc = 0; train_n = 0; i = 0
    if args.exact_mode == "SA":
        # change the optimizer schedule
        opt = optim.Adam(model.parameters(), lr=0.1)

    for batch_idx, (data, target) in enumerate(loader):
        X, y = data.to(args.device), target.to(args.device)
        yp = model(X)

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

    return train_loss / train_n, train_acc / train_n


# for testing
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


# python model_train.py --data_name $data_name --exact_mode 'teacher' --train_epochs 25
# python model_train.py --data_name $data_name --exact_mode 'fine-tune' --train_epochs 10
# python model_train.py --data_name $data_name --exact_mode 'retrain' --train_epochs 20
# python model_train.py --data_name $data_name --exact_mode 'prune' --train_epochs 20
# python model_train.py --data_name $data_name --exact_mode 'SA' --train_epochs 25
# python model_train.py --data_name $data_name --exact_mode 'DA-LENET' --train_epochs 25
# python model_train.py --data_name $data_name --exact_mode 'DA-VGG' --train_epochs 25
# *** main function of training ***/
def train(args):

    # 1. load the xx
    train_loader, test_loader = data_init(args.data_name, args.batch_size)

    # 2. load the model
    model = model_init(args.data_name, args.exact_mode, args.device, mode="MODEL")

    # 3. model training
    print("Model training...")
    args.lr_min = 0.; args.lr_max = 0.1
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    lr_schedule = lambda t: \
    np.interp([t], [0, args.train_epochs * 2 // 5, args.train_epochs * 4 // 5, args.train_epochs],
              [args.lr_min, args.lr_max, args.lr_max / 10, args.lr_min])[0]

    for t in range(args.train_epochs):
        lr = lr_schedule(t)
        model.train()
        train_loss, train_acc = train_epoch(args, train_loader, model, lr_schedule=lr_schedule, epoch_i=t, opt=optimizer)
        model.eval()
        test_loss, test_acc = test_epoch(args, test_loader, model)
        print(f'Epoch: {t}, Train Loss: {train_loss:.3f} Train Acc: {train_acc:.3f} Test Acc: {test_acc:.3f} lr: {lr:.3f}')

    torch.save(model.state_dict(), f"{args.model_dir}/final.pt")


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
