import os
import time
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import re
from model_data_init import data_init, model_init
from model_train import test_epoch
from params import args_parse
from adv_generation import set_target, adv_attack
from utils import *

def get_data(args, loader, mode):
    x_list = []
    i_list = []
    for i in range(args.num_class):
        x_list.append([])
        i_list.append([])

    # get center point of each class
    for idb, (XX_tgt, YY_tgt) in enumerate(loader):
        for idx in range(len(YY_tgt)):
            yy_tgt = YY_tgt[idx]
            x_list[yy_tgt.item()].append(XX_tgt[idx])

    x_center = torch.zeros(args.num_class, 3, 64, 64)
    for class_idx in range(args.num_class):
        l = x_list[class_idx]
        list_d = torch.zeros(len(l))

        for idl in tqdm(range(len(l))):
            sum_d = torch.tensor(0.)
            for jdl in range(len(l)):
                sum_d += torch.norm(l[idl] - l[jdl], p=2)
            list_d[idl] = sum_d

        mdx = int(torch.argmax(list_d))
        x = l[mdx]
        x_center[class_idx] = x
        # print('class:{}, list:{}, mdx:{}'.format(class_idx, list_d, mdx))

    x_center = np.array(x_center)
    np.set_printoptions(threshold=np.inf)
    # print(x_center)
    np.save(f"{args.point_dir}/{mode}_center_point.npy", x_center)


def get_embeddings_black(args, x_list, loader, model):
    print("Generating MinAD_KRM embeddings...")

    # get adversarial target samples
    tgt_list = set_target(model, loader, args.num_class, args.device)

    distance_list = torch.zeros(args.num_class, 5, 3, device=args.device)
    adv_list = torch.zeros(args.num_class, 5, 3, 64, 64, device=args.device)

    for yy in range(len(x_list)):
        xx = torch.from_numpy(x_list[yy])
        xx = xx.unsqueeze(0).to(args.device)

        yy = torch.tensor([yy]).to(args.device)

        start = time.time()

        for class_idx in range(args.num_class):
            print(class_idx,"cccidx")
            # if args.data_name == "CIFAR100":
            #     preds = model(xx).squeeze(0)
            #     adv_idx = torch.argsort(preds, descending=True)[class_idx + 1].item()
            # else:
            #     adv_idx = class_idx
            adv_idx = class_idx
            adv_label = torch.tensor([adv_idx], device=args.device)
            if adv_label == yy:
                continue

            tgt = tgt_list[adv_idx]

            # logger.info('MinAD KRM process|| yy_label:{}, adv_label:{}, distance:{}'.
            #             format(yy, adv_label, torch.norm(tgt - xx, p=2)))

            # MinAD begins
            # print(tgt,adv_label,yy,"yyy")
            adv, d = adv_attack(args, model, xx, yy, tgt, adv_label)
            distance_list[yy, class_idx] = d
            adv_list[yy, class_idx] = adv.squeeze(0)
            print(class_idx,"classidx")
        print('MinAD KRM|| yy_label:{}, distance:{}, t:{}'.format(yy, distance_list[yy], time.time() - start))

    full_d = distance_list
    full_adv = adv_list

    print(full_d.shape)
    print(adv_list.shape)

    return full_d, full_adv


def get_init(childname):
    filename = childname # 使用正则表达式提取第一个 .pth 之前的部分
    match = re.search(r"^(.*?)(?=\.pth)", filename)
    if match:   
        parentindex = match.group(1)
        # print(extracted_string)  # 输出 'seed420'
    else:       
        parentindex=None
    parentname=parentindex+'.pth'
    parentpath='./../../modelscls5/parents'
    initpath='./../../modelscls5/init'
    childpath='./../../modelscls5/childs'
    return parentindex
# python knowledge_matrix.py --data_name $data_name --feature_mode 'MinAD_KRM'
# *** main function for knowledge representation ***/
def get_knowledge(args,modelname):

    # 1. load the xx
    # train_loader, test_loader = data_init(args.data_name, args.batch_size)
    # index=0
    index=get_init(modelname)
    batch_size=64
    train_loader,test_loader=get_cifar100_loader(int(index), batch_size)

    # 2. load targeted model
    # model = model_init(args.data_name, args.exact_mode, args.device, mode="FEATURE")
    # model = nn.DataParallel(model).to(args.device)
    # location = f"{args.model_dir}/final.pt"
    model=SimpleCNN()
    parentpath='./../../../modelscls5/parents'
    location=os.path.join(parentpath,modelname)
    model.load_state_dict(torch.load(location, map_location=args.device))
    model=model.to('cuda:0')

    _, test_acc = test_epoch(args, test_loader, model)
    print(f'Model: {args.model_dir} | Test Acc: {test_acc:.3f}')

    # 3. get center point of each class
    if not os.path.exists(f"{args.point_dir}/train_center_point.npy"):
        get_data(args, train_loader, mode="train")
    if not os.path.exists(f"{args.point_dir}/test_center_point.npy"):
        get_data(args, test_loader, mode="test")

    x_train = np.load(f"{args.point_dir}/train_center_point.npy")
    x_test = np.load(f"{args.point_dir}/test_center_point.npy")

    # 4. adversarial example generation
    # train_d, train_adv = get_embeddings_black(args, x_train, train_loader, model)
    # print("train embeddings:", train_d)
    # torch.save(train_d, f"{args.feature_dir}/train_{args.feature_mode}_features.pt")
    # torch.save(train_adv, f"{args.feature_dir}/train_{args.feature_mode}_advs.pt")

    test_d, test_adv = get_embeddings_black(args, x_test, test_loader, model)
    print("test embeddings:", test_d)
    torch.save(test_d, f"{args.feature_dir}/test_{args.feature_mode}_features.pt")
    torch.save(test_adv, f"{args.feature_dir}/test_{args.feature_mode}_advs.pt")


if __name__ == "__main__":

    parser = args_parse()
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.cuda.set_device(device)
    print(device)

    # set model saving path
    model_paths="D:\项目\SZY\pythonProject1\MIATB\SZY2025\modelscls5/parents/"
    modelnames=os.listdir(model_paths)

    for modelname in modelnames:
        if os.path.exists(f"./szyfeatures/train_{args.feature_mode}_advs.pt"):
            continue
        model_dir=os.path.join(model_paths,modelname)
        # model_dir=f"D:\项目\SZY\pythonProject1\MIATB\SZY2025\modelscls5/childs/0.pthdataset0.pth"
        # model_dir = f"./models/{args.data_name}/model_{args.exact_mode}"
        args.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Model Directory:", model_dir)
        # set feature saving path
        # feature_dir = f"./szyfeatures/model_{args.exact_mode}"
        feature_dir=os.path.join("./szyfeatures/", modelname)
        args.feature_dir = feature_dir
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        print("Feature Directory:", feature_dir)

        # set xx saving path
        point_dir = f"./points/"+modelname
        args.point_dir = point_dir
        if not os.path.exists(point_dir):
            os.makedirs(point_dir)
        print("Points Directory:", point_dir)

        # n_class = {"CIFAR10": 10, "CIFAR100": 100}
        # args.num_class = n_class[args.data_name]
        args.num_class=5
        torch.manual_seed(args.seed)

        get_knowledge(args,modelname)

