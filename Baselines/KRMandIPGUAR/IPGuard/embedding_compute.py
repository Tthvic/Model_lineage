import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
    # vector of original data and adversarial sample
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


def binary_search(args, model, x_0, x_random, adv_label, tol=1e-2):
    """
    * Optimized binary search function to project adversarial sample onto the decision boundary
    :param args: parameters
    :param model: target model
    :param x_0: target sample
    :param x_random: adversarial sample $\tilde{x}$
    :param adv_label: adversarial label yk
    :param tol: the threshold value
    :return adv: adversarial sample $\bar{x}$ at the boundary
    """
    adv = x_random.detach().clone()
    cln = x_0.detach().clone()

    # Ensure all tensors are on the same device
    device = x_0.device
    adv = adv.to(device)
    cln = cln.to(device)

    while True:
        # Compute mid-point in-place to reduce memory overhead
        mid = ((cln + adv) / 2.0).to(device)

        # Check decision function
        if decision_function(args, model, mid, adv_label):
            adv = mid
        else:
            cln = mid

        # Convergence condition (avoid CPU-GPU transfer)
        if torch.norm(adv - cln) < tol:
            break

    return adv.detach()



def decision_function(args, model, images, adv_label, batch_size=100):
    """
    Decision function output 1 on the desired side of the boundary, 0 otherwise
    :param args: parameters
    :param model: target model
    :param images: noise sample
    :param adv_label: adversarial label yk
    :param batch_size: the size of a batch for reducing time of prediction
    :return: whether the data point is adversarial (1:yes 0:no)
    """
    model.train(mode=False)

    # the prediction is made directly if there is only one data,
    if len(images) == 1:
        with torch.no_grad():
            predict_label = torch.argmax(model(images), dim=1).reshape(len(images), 1)
            # top2_values, top2_indices = torch.topk(model(images), k=2, dim=1)
            # print(top2_values,top2_indices,"tttindices")
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


def adv_break_flag(image, model, ori_label, tar_label, threshold=0.1):
    """
    判断是否需要提前停止攻击。
    :param image: 输入图像张量 (batch_size, ...)
    :param model: 目标模型
    :param ori_label: 原始标签
    :param tar_label: 目标标签
    :param threshold: 两个预测值之间的允许差距阈值
    :return: 如果满足条件返回 True，否则返回 False
    """
    # 获取模型预测结果的 top-2 预测值和对应索引
    top2_values, top2_indices = torch.topk(model(image), k=2, dim=1)
    # 将 top-2 索引转换为列表以便比较
    top2_indices = top2_indices.squeeze()  # 移除多余的维度
    # 检查 top-2 索引是否包含 ori_label 和 tar_label
    # print(top2_indices,ori_label,tar_label,"llabel",set(top2_indices.tolist()),  {ori_label, tar_label},
          # set(top2_indices.tolist()) == {ori_label, tar_label})
    if top2_indices.tolist()==[ori_label,tar_label] or top2_indices.tolist()==[tar_label,ori_label]:
        # 计算两者的预测值差距
        value_diff = abs(top2_values[0, 0].item() - top2_values[0, 1].item())
        # 判断差距是否小于阈值
        # print(value_diff,"value_difff")
        if value_diff < threshold:
            return True
    # 如果不满足条件，返回 False
    return False

def adv_attack(args, model, x_sample, y_sample, tgt_sample, adv_label):
    """
    MinAD function for adversarial attack.
    """
    # Step 1: Get the initial adversarial sample at the boundary
    adv_init = binary_search(args, model, x_sample, tgt_sample, adv_label)
    adv_update = adv_init.detach().clone()
    ori_sample = x_sample.detach().clone()

    # Compute initial distance
    distance_init = torch.norm(adv_init - ori_sample, p=2)
    distance_value = distance_init

    # Step 2: Main loop for updating the adversarial sample
    lr1 = args.attack_lr_begin
    temp_list = []

    for attack_epoch in range(args.num_attack_iter):
        # Clone the current adversarial sample
        adv_sample = adv_update.detach().clone()
        distance = distance_value.detach().clone()

        # Estimate normal vector
        adv_sample.requires_grad = True
        loss = loss_cos(args, model, adv_sample, ori_sample, adv_label, distance)
        loss.backward()
        grads = adv_sample.grad / torch.norm(adv_sample.grad, p=2)
        adv_sample.requires_grad = False

        # Update the sample
        lr2 = lr1
        temp = 0
        adv_sample -= lr1 * grads

        # Ensure the updated sample is adversarial
        while not decision_function(args, model, adv_sample, adv_label):
            temp = 1
            lr2 /= 2
            adv_sample += lr2 * grads
            if lr2 < args.attack_lr_end / 2:
                break

        # Append temporary flag for learning rate adjustment
        temp_list.append(temp)

        # Adjust learning rate
        if not temp and lr1 > args.attack_lr_end or sum(temp_list[max(0, attack_epoch - 4):attack_epoch]) == 4:
            lr1 /= 2
            if lr1 < args.attack_lr_end:
                lr1 = args.attack_lr_end

        # Project the updated sample back to the decision boundary
        with torch.no_grad():
            adv_sample = binary_search(args, model, ori_sample, adv_sample, adv_label)
            distance_value = torch.norm(adv_sample - ori_sample, p=2)

        # Update the adversarial sample
        adv_update = adv_sample.detach().clone()

        # Check early stopping condition
        if adv_break_flag(adv_sample, model, y_sample, adv_label):
            # print("Break")
            break

        # # Optional: Print progress
        # if args.verbose and attack_epoch % 10 == 0:
        #     print("MinAD: epoch:{}, adv_label:{}, loss:{}, distance:{}".format(
        #         attack_epoch, torch.argmax(model(adv_sample)), loss.item(), distance_value.item()))

    # Step 3: Finalize results
    print("MinAD: ori_sample:{}, adv_label:{}, distance:{}".format(y_sample, adv_label, distance_value))
    return adv_update.detach().clone(), torch.tensor([distance_value.item()], device=args.device)
    # return adv_sample, torch.tensor([d_linf, d_l2, d_l1], device=args.device)


def get_black_vulnerabilty(args, loader, model):
    print("Getting MinAD embeddings...")

    data_num = 0

    # set target adversarial sample list
    tgt_list = set_target(model, loader, args.num_class, args.device)

    # define a matrix that stores distances
    distance_list = torch.zeros(args.num_samples, 10, 3, device=args.device)
    # define a matrix that stores adversarial examples
    adv_list = torch.zeros(args.num_samples, 10, 3, 32, 32, device=args.device)

    for idb, (XX_tgt, YY_tgt) in enumerate(loader):

        for idx in range(args.batch_size):

            # limit number of samples
            if data_num >= args.num_samples:
                break

            # inputs settings
            xx_tgt = XX_tgt[idx].unsqueeze(0).to(args.device)
            yy_tgt = YY_tgt[idx].unsqueeze(0).to(args.device)

            # make sure the model prediction is correct
            if yy_tgt != torch.argmax(model(xx_tgt), -1):
                continue

            start = time.time()

            for class_idx in range(10):

                if args.data_name == "CIFAR100":
                    preds = model(xx_tgt).squeeze(0)
                    adv_idx = torch.argsort(preds, descending=True)[class_idx + 1].item()
                else:
                    adv_idx = class_idx

                adv_label = torch.tensor([adv_idx], device=args.device)
                if adv_label == yy_tgt:
                    continue

                tgt_tgt = tgt_list[adv_idx]

                print('MinAD attack process|| data_num:{}, yy_label:{}, adv_label:{}, distance:{}'.
                      format(data_num, yy_tgt, adv_label, torch.norm(tgt_tgt - xx_tgt, p=2)))

                adv, d = adv_attack(args, model, xx_tgt, yy_tgt, tgt_tgt, adv_label)
                distance_list[data_num, class_idx] = d
                adv_list[data_num, class_idx] = adv.squeeze(0)

            print('MinAD attack|| data_num:{}, distance:{}, t:{}'.format(data_num, distance_list[data_num], time.time() - start))

            data_num += 1

    full_d = distance_list

    print(full_d.shape)

    return full_d


def set_target(model, loader, num_class, device):
    ## Find the corresponding adversarial target sample($\tilde{x}_0$) for each class
    tgt_list = []
    flag=1
    print(num_class,"num_class")
    for tgt_label in range(num_class):
        tgt_label = torch.tensor(tgt_label)
        tgt_temp = 0

        for idb, (XX_tgt, YY_tgt) in enumerate(loader):
            if tgt_temp:
                break
            # if idb>2:
                # break
            for i_yy in range(len(YY_tgt)):
                # find the sample
                orilabel=torch.argmax(model(XX_tgt[i_yy].unsqueeze(0).to(device)))
                if tgt_label == orilabel.cpu():
                    tgt_tgt = XX_tgt[i_yy].unsqueeze(0).to(device)
                    # add the sample
                    tgt_list.append(tgt_tgt)
                    tgt_temp = 1
                    break
        if tgt_temp==0:
            tgt_list.append(torch.zeros([1,3,64,64]))
    # print(num_class,tgt_label,len(tgt_list),"tgt_class1")
    if len(tgt_list) == num_class:
        flag=0

    return tgt_list,flag

# python embedding_compute.py --data_name $data_name --exact_mode $exact_mode --feature_mode 'MinAD'
# *** main function for knowledge embedding generation ***/
def embeddings_extractor(args):

    # 1. load the data
    train_loader, test_loader = data_init(args.data_name, args.batch_size, pseudo_labels=False, train_shuffle=False)

    # 2. load the student model
    _, student = model_init(args.data_name, args.exact_mode, args.num_class, args.pseudo_labels, args.device, temp='feature')
    student = nn.DataParallel(student).to(args.device)
    location = f"{args.model_dir}/final.pt"
    student.load_state_dict(torch.load(location, map_location=args.device))

    student.eval()

    _, test_acc = test_epoch(args, test_loader, student)
    print(f'Model: {args.model_dir} | \t Test Acc: {test_acc:.3f}')

    # 3. generate embeddings
    train_d = get_black_vulnerabilty(args, train_loader, student)
    print("train embeddings:", train_d)
    torch.save(train_d, f"{args.feature_dir}/train_{args.feature_mode}_features.pt")

    test_d = get_black_vulnerabilty(args, test_loader, student)
    print("test embeddings:", test_d)
    torch.save(test_d, f"{args.feature_dir}/test_{args.feature_mode}_features.pt")


if __name__ == "__main__":

    parser = args_parse()
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.cuda.set_device(device)
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

    n_class = {"CIFAR10": 10, "CIFAR100": 100}
    args.num_class = n_class[args.data_name]

    torch.manual_seed(args.seed)

    embeddings_extractor(args)

