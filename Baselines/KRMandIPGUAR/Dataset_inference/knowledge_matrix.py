import os
import time
import numpy as np
import torch
print(torch.version.cuda,"versioncuda")
import torch.nn as nn

from model_data_init import data_init, model_init
from model_train import test_epoch
from params import args_parse
from embedding_compute import set_target, adv_attack


def get_data(args, loader, mode, temp):
    x_list = []
    i_list = []
    for i in range(args.num_class):
        x_list.append([])
        i_list.append([])

    for idb, (XX_tgt, YY_tgt) in enumerate(loader):
        for idx in range(len(YY_tgt)):
            yy_tgt = YY_tgt[idx]
            x_list[yy_tgt.item()].append(XX_tgt[idx])

    # generate the center point of each class
    if temp == 'center':
        x_center = torch.zeros(args.num_class, 3, 32, 32)
        for class_idx in range(args.num_class):
            l = x_list[class_idx]
            list_d = torch.zeros(len(l))

            for idl in range(len(l)):
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
        print(x_center)
        np.save(f"{args.point_dir}/{mode}_center_point.npy", x_center)

    # generate the sub-center point of each class
    if temp == 'sub-center':
        for idx_center in range(4):
            for class_idx in range(args.num_class):
                l = x_list[class_idx]
                i_list[class_idx] = l[int(len(l)*idx_center/4): int(len(l)*(idx_center+1)/4)]

            # find the center point of each part
            x_center = torch.zeros(args.num_class, 3, 32, 32)
            for class_idx in range(args.num_class):
                l = i_list[class_idx]
                list_d = torch.zeros(len(l))

                for idl in range(len(l)):
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
            print(idx_center, x_center)
            np.save(f"{args.point_dir}/{mode}_center{idx_center+1}_point.npy", x_center)


def get_embeddings_black(args, x_list, loader, model):
    print("Getting MinAD_KRM embeddings...")

    # set target adversarial sample list
    tgt_list = set_target(model, loader, args.num_class, args.device)

    # define a matrix that stores distances
    distance_list = torch.zeros(args.num_class, 10, 3, device=args.device)
    # define a matrix that stores adversarial examples
    adv_list = torch.zeros(args.num_class, 10, 3, 32, 32, device=args.device)

    for yy in range(len(x_list)):

        xx = torch.from_numpy(x_list[yy])
        xx = xx.unsqueeze(0).to(args.device)

        yy = torch.tensor([yy]).to(args.device)

        start = time.time()

        for class_idx in range(10):

            if args.data_name == "CIFAR100":
                preds = model(xx).squeeze(0)
                adv_idx = torch.argsort(preds, descending=True)[class_idx + 1].item()
            else:
                adv_idx = class_idx

            adv_label = torch.tensor([adv_idx], device=args.device)
            if adv_label == yy:
                continue

            tgt = tgt_list[adv_idx]

            # logger.info('MinAD KRM process|| yy_label:{}, adv_label:{}, distance:{}'.
            #             format(yy, adv_label, torch.norm(tgt - xx, p=2)))

            # model->目标模型 xx->目标数据输入 yy->目标数据标签, tgt_tgt->目标攻击初始对抗数据
            adv, d = adv_attack(args, model, xx, yy, tgt, adv_label)
            distance_list[yy, class_idx] = d
            adv_list[yy, class_idx] = adv.squeeze(0)

        print('MinAD KRM|| yy_label:{}, distance:{}, t:{}'.format(yy, distance_list[yy], time.time() - start))

    full_d = distance_list

    print(full_d.shape)

    return full_d


# python knowledge_matrix.py --data_name $data_name --exact_mode $exact_mode --feature_mode 'MinAD_KRM'
# *** main function for knowledge representation ***/
def get_knowledge(args):

    # 1. load the xx
    train_loader, test_loader = data_init(args.data_name, args.batch_size, pseudo_labels=False, train_shuffle=False)

    # 2. load the student model
    _, student = model_init(args.data_name, args.exact_mode, args.num_class, args.pseudo_labels, args.device,
                            temp='feature')
    student = nn.DataParallel(student).to(args.device)
    location = f"{args.model_dir}/final.pt"
    student.load_state_dict(torch.load(location, map_location=args.device))

    student.eval()

    _, test_acc = test_epoch(args, test_loader, student)
    print(f'Model: {args.model_dir} | Test Acc: {test_acc:.3f}')

    # 3. get and load center points and sub-center points(if need)
    if not os.path.exists(f"{args.point_dir}/train_center_point.npy"):
        get_data(args, train_loader, mode="train", temp="center")
    if not os.path.exists(f"{args.point_dir}/test_center_point.npy"):
        get_data(args, test_loader, mode="test", temp="center")

    num_center = args.num_center
    for idx_center in range(num_center - 1):
        if not os.path.exists(f"{args.point_dir}/train_center{idx_center+1}_point.npy"):
            get_data(args, train_loader, mode="train", temp="sub-center")
        if not os.path.exists(f"{args.point_dir}/test_center{idx_center+1}_point.npy"):
            get_data(args, test_loader, mode="test", temp="sub-center")

    x_train = np.load(f"{args.point_dir}/train_center_point.npy")
    x_test = np.load(f"{args.point_dir}/test_center_point.npy")
    for idx_center in range(num_center - 1):
        x_tr = np.load(f"{args.point_dir}/train_center{idx_center+1}_point.npy")
        x_train = np.vstack([x_train, x_tr])
        x_te = np.load(f"{args.point_dir}/test_center{idx_center+1}_point.npy")
        x_test = np.vstack([x_test, x_te])

    # 4. generate embeddings
    train_d = get_embeddings_black(args, x_train, train_loader, student)
    print("train embeddings:", train_d)
    torch.save(train_d, f"{args.feature_dir}/train_{args.feature_mode}{args.num_center}_features.pt")

    test_d = get_embeddings_black(args, x_test, test_loader, student)
    print("test embeddings:", test_d)
    torch.save(test_d, f"{args.feature_dir}/test_{args.feature_mode}{args.num_center}_features.pt")


if __name__ == "__main__":

    parser = args_parse()
    args = parser.parse_args()
    print(args)
    device='cpu'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # torch.cuda.set_device(device)
    print(device)

    # set model saving path
    model_dir = f"./models/{args.data_name}/model_{args.exact_mode}"
    args.model_dir = model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print("Model Directory:", model_dir)

    # set embeddings saving path
    feature_dir = f"./features/{args.data_name}/model_{args.exact_mode}"
    args.feature_dir = feature_dir
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    print("Feature Directory:", feature_dir)

    # set points saving path
    point_dir = f"./points/{args.data_name}"
    args.point_dir = point_dir
    if not os.path.exists(point_dir):
        os.makedirs(point_dir)
    print("Points Directory:", point_dir)

    n_class = {"CIFAR10": 10, "CIFAR100": 100}
    args.num_class = n_class[args.data_name]

    torch.manual_seed(args.seed)

    get_knowledge(args)
