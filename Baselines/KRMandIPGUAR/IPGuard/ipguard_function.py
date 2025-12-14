import random
import numpy as np
import torch
import torch.nn as nn
from params import args_parse
from model_data_init import data_init, model_init
from model_train import test_epoch


# python ipguard_function.py --data_name $data_name --feature_mode $feature_mode
# *** main function of IPGuard ***/
def ipguard_function(args):

    # suspected models
    exact_names = ["teacher", "fine-tune", "retrain", "prune", "SA", "DA-LENET", "DA-VGG"]

    ## 1. load the xx
    train_loader, test_loader = data_init(args.data_name, args.batch_size)

    ## 2. load suspected models
    model = {}
    for name in exact_names:
        args.exact_mode = name
        model_dir = f"./models/{args.data_name}/model_{args.exact_mode}"

        student = model_init(args.data_name, args.exact_mode, args.device, mode='FEATURE')
        student = nn.DataParallel(student).to(args.device)
        location = f"{model_dir}/final.pt"
        student.load_state_dict(torch.load(location, map_location=args.device))

        student.eval()

        model[name] = student

        # testing
        for name in exact_names:
            _, test_acc = test_epoch(args, test_loader, model[name])
            print(f'{name} Model Test || Test Acc: {test_acc:.3f}')

    ## 3. load adversarial examples
    feature_dir = f"./features/{args.data_name}/model_teacher"
    train_data = torch.load(f"{feature_dir}/train_{args.feature_mode}_advs.pt")
    test_data = torch.load(f"{feature_dir}/test_{args.feature_mode}_advs.pt")
    data = torch.cat((train_data, test_data), dim=0)
    if args.feature_mode == 'MinAD_KRM':
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4])

    ## 4. IPGuard
    result = {}
    for name in exact_names:
        result[name] = []

    split_index = random.sample(range(len(data)), args.i_num)
    data_iptest = data[split_index]

    # prediction of teacher model
    target_model = model['teacher']
    target_predict = torch.argmax(target_model(data_iptest), dim=1)
        
    # prediction of suspected model
    for name in exact_names:
        suspect_model = model[name]
        suspect_predict = torch.argmax(suspect_model(data_iptest), dim=1)

        # compute matching rate
        match_rate = torch.sum(torch.where(target_predict == suspect_predict, 1, 0)) / target_predict.shape[0]

        result[name].append(match_rate.item())


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

    ipguard_function(args)