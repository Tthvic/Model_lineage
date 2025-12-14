import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import ttest_ind

from params import args_parse


def feature_init(args, names):

    feature_dir = f"./features/{args.data_name}"

    trains = {}
    tests = {}
    for name in names:
        trains[name] = torch.load(f"{feature_dir}/model_{name}/train_{args.feature_mode}_features.pt")
        tests[name] = torch.load(f"{feature_dir}/model_{name}/test_{args.feature_mode}_features.pt")

    if args.feature_mode == 'MinAD_KRM':
        # load more than one knowledge representation matrix
        for name in names:
            trains[name] = torch.load(f"{feature_dir}/model_{name}/train_{args.feature_mode}{args.num_center}_features.pt").to(args.device)
            tests[name] = torch.load(f"{feature_dir}/model_{name}/test_{args.feature_mode}{args.num_center}_features.pt").to(args.device)

        trains_n = {}
        tests_n = {}
        for name in names:
            trains_n[name] = trains[name].T.reshape(
                trains[name].shape[0] * trains[name].shape[1], trains[name].shape[2]).cpu()
            tests_n[name] = tests[name].T.reshape(
                tests[name].shape[0] * tests[name].shape[1], trains[name].shape[2]).cpu()

        # get normalized
        mean_cifar = trains_n["teacher"].mean(dim=0)
        std_cifar = trains_n["teacher"].std(dim=0)

        for name in names:
            trains_n[name] = (trains_n[name] - mean_cifar) / std_cifar
            trains_n[name] = (trains_n[name] - mean_cifar) / std_cifar

    elif args.feature_mode == 'MinAD':
        mean_cifar = trains["teacher"].mean(dim=(0, 1))
        std_cifar = trains["teacher"].std(dim=(0, 1))

        for name in names:
            trains[name] = trains[name].sort(dim=1)[0]
            tests[name] = tests[name].sort(dim=1)[0]

        for name in names:
            trains[name] = (trains[name] - mean_cifar) / std_cifar
            tests[name] = (tests[name] - mean_cifar) / std_cifar

        trains_n = {}
        tests_n = {}
        args.f_num = trains['teacher'].shape[1] * trains['teacher'].shape[2]
        for name in names:
            trains_n[name] = trains[name].T.reshape(trains[name].shape[0], args.f_num).cpu()
            tests_n[name] = tests[name].T.reshape(tests[name].shape[0], args.f_num).cpu()

    else:
        raise Exception('Unknown feature mode')

    return trains_n, tests_n


# train confidence regressor gV
def regressor_train(args, trains_n, tests_n):

    if args.feature_mode == 'MinAD_KRM':
        # not limit the number of samples if with KRM
        args.num_samples = trains_n["teacher"].shape[0]
        args.f_num = 3

    train = torch.cat((trains_n["teacher"], tests_n["teacher"]), dim=0)
    y = torch.cat((torch.zeros(args.num_samples), torch.ones(args.num_samples)), dim=0)
    # print('train_shape:', train.shape)

    rand = torch.randperm(y.shape[0])
    train = train[rand]
    y = y[rand]

    model = nn.Sequential(nn.Linear(args.f_num, 100), nn.ReLU(), nn.Linear(100, 1), nn.Tanh())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # training settings
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = train
        outputs = model(inputs)
        loss = -1 * ((2*y-1)*(outputs.squeeze(-1))).mean()
        loss.backward()
        optimizer.step()
        if not (epoch+1) % 100:
            print('epoch {} loss {}'.format(epoch, loss.item()))

    return model


# for test
def inference_test(args, names, outputs_tr, outputs_te, result):

    def get_p(outputs_train, outputs_test):
        pred_test = outputs_test[:, 0].detach().cpu().numpy()
        pred_train = outputs_train[:, 0].detach().cpu().numpy()
        tval, pval = ttest_ind(pred_test, pred_train, alternative="greater", equal_var=False)
        if pval < 0:
            raise Exception(f"p-value={pval}")
        return pval

    def print_inference(outputs_train, outputs_test):
        m1, m2 = outputs_test[:, 0].mean(), outputs_train[:, 0].mean()
        pval = get_p(outputs_train, outputs_test)
        return pval, m1 - m2

    for name in names:
        p, m = print_inference(outputs_tr[name], outputs_te[name])
        result[name].append([p.item(), m.item()])

    return result


# python dataset_inference.py --data_name $data_name --feature_mode $feature_mode
# *** main function for knowledge representation ***/
def dataset_inference(args):

    # suspected models
    exact_names = ['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher']

    # 1. initialize the embeddings
    trains_n, tests_n = feature_init(args, exact_names)

    # 2. train confidence regressor gV
    model = regressor_train(args, trains_n, tests_n)

    result = {}
    for name in exact_names:
        result[name] = []

    # 3. get model output of each suspected model
    outputs_tr = {}
    outputs_te = {}
    for name in exact_names:
        split_index = random.sample(range(len(trains_n[name])), args.a_num)
        outputs_tr[name] = model(trains_n[name][split_index])
        outputs_te[name] = model(tests_n[name][split_index])

    # 4. the test for dataset inference
    result = inference_test(args, exact_names, outputs_tr, outputs_te, result)
    print('result:', result)


if __name__ == "__main__":

    parser = args_parse()
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    torch.cuda.set_device(device)
    print(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_inference(args)




