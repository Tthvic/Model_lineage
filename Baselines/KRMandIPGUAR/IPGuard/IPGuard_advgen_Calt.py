import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from model_data_init import data_init, model_init
from model_train import test_epoch
from params import args_parse
from utils import *
from caltech101 import *
import re
dimdict={'Calt_p':6,'Calt_c':4,'Dogs_p':8,'Dogs_c':3,'Flowers_p':6,'Flowers_c':2,'Calt_s':2,
         'Dogs_s':2, 'Flowers_s':2, 'Aircraft_cs':3,'Pet_cs':3,'Calt_cs':3}


def extract_number_from_filename(filename):
    # 使用正则表达式匹配文件名中的数字
    match = re.search(r'_(\d+)\.', filename)
    if match:
        # 提取并返回匹配的数字
        return int(match.group(1))
    else:
        # 如果没有找到匹配项，则抛出异常或返回一个默认值
        raise ValueError(f"Filename {filename} does not contain a number enclosed by '_' and '.'")


def load_pmodels(pname):
    pmodelpath='./../../../szy2025/train/train_res18/NONIMAGENETPmodels'
    modelpath=os.path.join(pmodelpath,pname)
    newdict=torch.load(modelpath)

    dim1 = 512
    dim2=dimdict['Calt_p']
    class Res18(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(Res18, self).__init__()
            self.module = torchvision.models.resnet18(pretrained=True)
            self.module.add_module('fc', torch.nn.Linear(dim1, dim2))

        def forward(self, x):
            x = self.module(x)
            return x

        def _feature_hook(self, module, input, output):
            # """
            # Hook function to capture the output of the last convolutional layer (before avgpool).
            # """
            self.registered_features = output

        def extract_features(self, x):
            """
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.avgpool.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features
    model = Res18(dim1=dim1, dim2=dim2).to(args.device)
    model.eval()
    model.load_state_dict(newdict)
    return model


def load_cmodels(pname):
    pmodelpath = './../../../szy2025/train/train_res18/NONCmodels'
    modelpath = os.path.join(pmodelpath, pname)
    newdict = torch.load(modelpath)

    dim1 = 512
    dim2 = dimdict['Calt_c']

    class Res18(torch.nn.Module):
        def __init__(self, dim1, dim2):
            super(Res18, self).__init__()
            self.module = torchvision.models.resnet18(pretrained=True)
            self.module.add_module('fc', torch.nn.Linear(dim1, dim2))

        def forward(self, x):
            x = self.module(x)
            return x

        def _feature_hook(self, module, input, output):
            # """
            # Hook function to capture the output of the last convolutional layer (before avgpool).
            # """
            self.registered_features = output

        def extract_features(self, x):
            """
            Extracts features from the input tensor before the classification layers.
            """
            # Register the hook on the last convolutional layer (layer4)
            handle = self.module.avgpool.register_forward_hook(self._feature_hook)
            # Perform the forward pass to trigger the hook
            self.forward(x)
            # Remove the hook after capturing the features
            handle.remove()
            return self.registered_features

    model = Res18(dim1=dim1, dim2=dim2).to(args.device)
    model.eval()
    model.load_state_dict(newdict)
    return model


def loss_cos(args, model, x_adv, x_ori, adv_label, distance):
    """
    * loss function
    :param args: parameters
    :param model: target model
    :param x_adv: adversarial sample at the boundary
    :param x_ori: target sample
    :param adv_label: adversarial label yk
    :param distance: distance of last step
    :return cos: [-1, 0]
    """
    # vector of original xx and adversarial sample
    x_sub = x_adv - x_ori
    x_sub = x_sub.view(-1)
    x_sub = x_sub / torch.norm(x_sub, p=2)

    # the normal vector of decision boundary on x_adv
    grad_xadv = gradient_compute_boundary(args, model, x_adv, adv_label, distance)
    grad_xadv = grad_xadv.view(-1)
    grad_xadv = grad_xadv / torch.norm(grad_xadv, p=2)

    # compute cosine similarity
    cos = -torch.cosine_similarity(x_sub, grad_xadv, dim=-1)

    return cos


def gradient_compute_boundary(args, model, sample, adv_label, distance):
    """
    * estimate the normal vector of decision boundary with MCMC
    :param args: parameters
    :param model: target model
    :param sample: adversarial sample at the boundary
    :param adv_label: adversarial label yk
    :param distance: distance of last step
    :return gradf: normal vector of decision boundary
    """
    # sample.shape = torch.size([1, 3, 32, 32])->torch.size([3, 32, 32])
    sample = sample.squeeze(0)
    # set number of disturbed groups B
    num_evals = args.num_evals_boundary
    # set step size $\delta$
    delta = 1 / len(sample.shape) * distance
    if delta > 0.2:
        delta = 0.2

    # get the noise
    noise_shape = [num_evals] + list(sample.shape)
    rv = torch.randn(*noise_shape, device=args.device)
    rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=(1, 2, 3), keepdim=True))

    # get the noise sample
    perturbed = sample + delta * rv

    # model prediction
    # compute $\Phi(\bar{x}+\delta ub)$
    decisions = decision_function(args, model, perturbed, adv_label)
    decision_shape = [len(decisions)] + [1] * len(sample.shape)
    fval = 2 * decisions.reshape(decision_shape) - 1.0

    # compute gradf as the normal vector N
    if torch.mean(fval) == torch.tensor(1.0):
        gradf = torch.mean(rv, dim=0)
    elif torch.mean(fval) == torch.tensor(-1.0):
        gradf = - torch.mean(rv, dim=0)
    else:
        fval -= torch.mean(fval)
        gradf = torch.mean(fval * rv, dim=0)

    return gradf


def binary_search(args, model, x_0, x_random, adv_label, tol=1e-5):
    """
    * binary search function to project adversarial sample onto the decision boundary
    :param args: parameters
    :param model: target model
    :param x_0: target sample
    :param x_random: adversarial sample $\tilde{x}$
    :param adv_label: adversarial label yk
    :param tol: the threshold value
    :return adv: adversarial sample $\bar{x}$ at the boundary
    """
    adv = x_random
    cln = x_0

    while True:

        mid = (cln + adv) / 2.0

        if decision_function(args, model, mid, adv_label):
            adv = mid
        else:
            cln = mid

        if torch.norm(adv - cln).cpu().numpy() < tol:
            break

    return adv


def decision_function(args, model, images, adv_label, batch_size=100):
    """
    Decision function output 1 on the desired side of the boundary, 0 otherwise
    :param args: parameters
    :param model: target model
    :param images: noise sample
    :param adv_label: adversarial label yk
    :param batch_size: the size of a batch for reducing time of prediction
    :return: whether the xx point is adversarial (1:yes 0:no)
    """
    model.train(mode=False)

    # the prediction is made directly if there is only one xx,
    if len(images) == 1:
        with torch.no_grad():
            predict_label = torch.argmax(model(images), dim=1).reshape(len(images), 1)

    else:
        results_list = []

        # Run prediction with batch processing
        num_batch = int(np.ceil(len(images) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, images.shape[0]),
            )
            with torch.no_grad():
                output = model(images[begin:end])
            output = output.detach().cpu().numpy().astype(np.float32)
            results_list.append(output)

        results = np.vstack(results_list)
        predict = torch.from_numpy(results).to(args.device)
        predict_label = torch.argmax(predict, dim=1).reshape(len(images), 1)

    target_label = torch.zeros((len(images), 1), device=args.device)
    for i in range((len(images))):
        target_label[i, 0] = adv_label

    return predict_label == target_label


def adv_attack(args, model, x_sample, y_sample, tgt_sample, adv_label):
    """
    MinAD function
    * 1.get the sample $\bar{xi}$ at the boundary by binary search;
    * 2.estimate normal vector of the boundary at $\bar{xi}$;
    * 3.update the sample $\tilde{xi+1}$ by SGD;
    * 4.get the sample $\bar{xi+1}$ at the boundary by binary search again.
    :param args: parameters
    :param model: target model
    :param x_sample: target sample x
    :param y_sample: label of x
    :param tgt_sample: initial adversarial sample $\tilde{x0}$
    :param adv_label: adversarial label yk
    :return adv_sample: the sample $\bar{xI}$ at the boundary
    :return distance: the distance between target sample x and model boundary
    """
    # get $\bar{x0}$ at the boundary
    adv_init = binary_search(args, model, x_sample, tgt_sample, adv_label)
    adv_update = adv_init
    ori_sample = x_sample

    distance_init = torch.norm(adv_init - ori_sample, p=2)
    distance_value = distance_init

    # main loop of update
    lr1 = args.attack_lr_begin
    temp_list = []
    for attack_epoch in range(args.num_attack_iter):

        adv_sample = copy.deepcopy(adv_update)
        distance = copy.deepcopy(distance_value)

        # estimate normal vector
        adv_sample.requires_grad = True
        loss = loss_cos(args, model, adv_sample, ori_sample, adv_label, distance)
        # logger.info("MinAD: epoch:{}, adv_label:{}, loss:{}, distance:{}".format(attack_epoch, torch.argmax(model(adv_sample)), loss, distance))
        loss.backward()
        grads = adv_sample.grad
        grads = grads / torch.norm(grads, p=2)
        adv_sample.requires_grad = False

        # update the sample
        # lr1: learning rate for update
        # lr2: we set a new learning rate to ensure that the updated xx is adversarial
        temp = 0
        adv_sample -= lr1 * grads

        # set the auxiliary debugging function: if the updated xx is not adversarial, the update step may be too large
        lr2 = lr1
        while not decision_function(args, model, adv_sample, adv_label):
            temp = 1
            lr2 = lr2 / 2
            adv_sample += lr2 * grads
            if lr2 < args.attack_lr_end / 2:
                break
        temp_list.append(temp)

        # decay of learning rate
        if not temp and lr1 > args.attack_lr_end \
                or sum(temp_list[attack_epoch-4: attack_epoch]) == 4:
            lr1 /= 2
            # set threshold
            if lr1 < args.attack_lr_end:
                lr1 = args.attack_lr_end

        # get the sample at the boundary by binary search
        adv_sample = binary_search(args, model, ori_sample, adv_sample, adv_label)

        # compute the distance between target sample x and final adversarial sample at the boundary
        distance_value = torch.norm(adv_sample - ori_sample, p=2)

        adv_update = copy.deepcopy(adv_sample)

    # logger.info("MinAD: ori_sample:{}, adv_label:{}, distance:{}".format(y_sample, adv_label, distance_value))
    d_linf = torch.norm(adv_sample - ori_sample, p=np.inf)
    d_l2 = distance_value
    d_l1 = torch.norm(adv_sample - ori_sample, p=1)

    return adv_sample, torch.tensor([d_linf, d_l2, d_l1], device=args.device)


def get_black_vulnerabilty(args, loader, model):
    print("Generating MinAD adversarial examples...")

    data_num = 0

    tgt_list = set_target(model, loader, args.num_class, args.device)

    distance_list = torch.zeros(args.num_samples, 3, device=args.device)
    adv_list = torch.zeros(args.num_samples, 3, 128, 128, device=args.device)

    for idb, (XX_tgt, YY_tgt) in enumerate(loader):

        for idx in range(args.batch_size):

            if data_num >= args.num_samples:
                break

            ## inputs setting
            # cifar10: X_ [64, 3, 32, 32]->[1, 3, 32, 32]
            # cifar10: Y_ [64]->[1]
            # cifar10: model(X)_ [1, 10]->[10, ]
            xx_tgt = XX_tgt[idx].unsqueeze(0).to(args.device)
            yy_tgt = YY_tgt[idx].unsqueeze(0).to(args.device)

            if yy_tgt != torch.argmax(model(xx_tgt), -1):
                continue

            start = time.time()

            # the nearest decision boundary is selected as the adversarial label
            yy2 = torch.argsort(model(xx_tgt).squeeze(0))[args.num_class - 2].item()
            adv_label = torch.tensor([yy2], device=args.device)

            tgt_tgt = tgt_list[yy2]

            # logger.info('MinAD process|| data_num:{}, yy_label:{}, adv_label:{}, distance:{}'.
            #           format(data_num, yy_tgt, adv_label, torch.norm(tgt_tgt - xx_tgt, p=2)))

            adv, d = adv_attack(args, model, xx_tgt, yy_tgt, tgt_tgt, adv_label)
            distance_list[data_num] = d
            adv_list[data_num] = adv.squeeze(0)

            print('MinAD|| data_num:{}, distance:{}, t:{}'.format(data_num, distance_list[data_num], time.time() - start))

            data_num += 1

    full_d = distance_list
    full_adv = adv_list

    print(full_d.shape)
    print(adv_list.shape)

    return full_d, full_adv


def set_target(model, loader, num_class, device):
    ## Find the corresponding adversarial target sample($\tilde{x}_0$) for each class

    tgt_list = []
    for tgt_label in range(num_class):
        tgt_label = torch.tensor(tgt_label)

        tgt_temp = 0
        for idb, (XX_tgt, YY_tgt) in enumerate(loader):
            if tgt_temp:
                break
            for i_yy in range(len(YY_tgt)):

                # find the sample
                if YY_tgt[i_yy] == tgt_label:
                    if tgt_label == torch.argmax(model(XX_tgt[i_yy].unsqueeze(0).to(device))).cpu():
                        tgt_temp = 1
                        tgt_tgt = XX_tgt[i_yy].unsqueeze(0).to(device)
                        # add the sample
                        tgt_list.append(tgt_tgt)
                        break

    assert len(tgt_list) == num_class

    return tgt_list


# python adversarial_generation.py --data_name $data_name --feature_mode 'MinAD'
# *** main function of adversarial example generation ***/
def embeddings_extractor(args,modelname):

    # 1. load the xx
    # train_loader, test_loader = data_init(args.data_name, args.batch_size)
    index=0
    batch_size=64
    # train_loader,test_loader=get_cifar100_loader(index, batch_size)
    # 2. load the student model
    # student=SimpleCNN()
    # student = model_init(args.data_name, args.exact_mode, args.device, mode="FEATURE")
    student=load_pmodels(modelname)

    student = student.to(args.device)
    location = f"{args.model_dir}"
    checkpoint=torch.load(location, map_location='cpu')
    student.load_state_dict(checkpoint)
    student.eval()
    print("loading student")
    select_class=oriclass[extract_number_from_filename(modelname)]
    test_loader,train_loader=get_data_loader(select_class)
    _, test_acc = test_epoch(args, test_loader, student)
    print(f'Model: {args.model_dir} | Test Acc: {test_acc:.3f}')

    # 3. adversarial example generation
    train_d, train_adv = get_black_vulnerabilty(args, train_loader, student)
    print("train embeddings:", train_d)
    # torch.save(train_d, f"{args.feature_dir}/train_{args.feature_mode}_features.pt")
    # torch.save(train_adv, f"{args.feature_dir}/train_{args.feature_mode}_advs.pt")
    torch.save(train_d,'./Calt/train_features.pt')
    torch.save(train_adv, './Calt/train_advs.pt' )
    test_d, test_adv = get_black_vulnerabilty(args, test_loader, student)
    torch.save(test_d,'./Calt/test_features.pt')
    torch.save(test_adv, './Calt/test_advs.pt' )
    # torch.save(test_d, f"{args.feature_dir}/test_{args.feature_mode}_features.pt")
    # torch.save(test_adv, f"{args.feature_dir}/test_{args.feature_mode}_advs.pt")


if __name__ == "__main__":

    parser = args_parse()
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.cuda.set_device(device)
    print(device)

    # set model saving path
    model_paths='./../../../szy2025/train/train_res18/NONIMAGENETPmodels'
    modelnames=os.listdir(model_paths)
    for modelname in modelnames:
        if not 'Calt' in modelname:
            continue
        model_dir=os.path.join(model_paths,modelname)
        args.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        print("Model Directory:", model_dir)
        # set feature saving path
        # feature_dir = f"./szyfeatures/model_{args.exact_mode}"
        feature_dir=os.path.join("./feas_Calt/", modelname)
        args.feature_dir = feature_dir
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        print("Feature Directory:", feature_dir)
        args.num_class=dimdict['Calt_p']
        torch.manual_seed(args.seed)
        print("adv")
        embeddings_extractor(args,modelname)