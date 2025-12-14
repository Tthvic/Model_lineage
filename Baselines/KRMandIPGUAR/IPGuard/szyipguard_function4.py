import random
import numpy as np
import torch
import torch.nn as nn
from params import args_parse
from model_data_init import data_init, model_init
from model_train import test_epoch
import os
from utils import *



def get_falsefather_name_and_index(filename):           
    # 找到第一个出现的 '.pth' 的位置
    pth_index = filename.find('.pth')
    print(pth_index,"pohindex")
    # 如果找到了 '.pth'，则提取并处理       
    if pth_index != -1:                             
        grandfather_index = filename[:pth_index]
        # 生成假的数据                  
        real_index = int(grandfather_index)
        fake_index = real_index + 1
        # 确保假的索引与真的不一样，并引入随机性
        if fake_index >= 19:  # 如果达到了或超过了19，选择一个小于19的随机数
            fake_index = random.randint(0, 18)
        # else:     
        #     fake_index += random.randint(0, 5)  # 增加一些随机偏移，但保持在合理范围内
        # 使用假的index重构grandfather_name
        fake_grandfather_name = f"{fake_index}.pth"
        
        return (fake_grandfather_name, fake_index)
    else:
        # 如果没有找到 '.pth'，也返回假的数据
        fake_index = random.randint(20, 30)  # 选择一个明显不同的随机数
        fake_grandfather_name = f"未找到_pth_后缀_fake_{fake_index}.pth"
        
        return (fake_grandfather_name, fake_index)
    
# python ipguard_function.py --data_name $data_name --feature_mode $feature_mode
# *** main function of IPGuard ***/
def ipguard_function(args,parentname):

    # suspected models
    # exact_names = ["teacher", "fine-tune", "retrain", "prune", "SA", "DA-LENET", "DA-VGG"]

    ## 1. load the xx
    # train_loader, test_loader = data_init(args.data_name, args.batch_size)

    ## 2. load suspected models
    model = {}
    ochildnames = [f"dataset{i}.pthdataset{j}.pth" for i in range(20) for j in range(20)]
    # ochildnames = [f"dataset{i}.pth" for i in range(20)]
    childspath='./../../../modelscls5/grandson'
    exact_names=[]
    for name in ochildnames:
        exactname=parentname+name
    #     args.exact_mode = name
    #     model_dir = f"./models/{args.data_name}/model_{args.exact_mode}"
        # student = model_init(args.data_name, args.exact_mode, args.device, mode='FEATURE')
        # student = nn.DataParallel(student).to(args.device)
        # location = f"{model_dir}/final.pt"
        location= os.path.join(childspath,exactname)
        if not os.path.exists(location):
            continue
        exact_names.append(exactname)
        student=SimpleCNN()
        student.load_state_dict(torch.load(location, map_location=args.device))
        student.eval()
        model[exactname] = student

        # testing
        # for name in exact_names:
        #     _, test_acc = test_epoch(args, test_loader, model[name])
        #     print(f'{name} Model Test || Test Acc: {test_acc:.3f}')

    ## 3. load adversarial examples
    # feature_dir = f"./features/{args.data_name}/model_teacher"
    # train_data = torch.load(f"{feature_dir}/train_{args.feature_mode}_advs.pt")
    # test_data = torch.load(f"{feature_dir}/test_{args.feature_mode}_advs.pt")
    # data = torch.cat((train_data, test_data), dim=0)
    feature_dir=f"./szyfeatures/"+parentname
    train_data = torch.load(f"{feature_dir}/train_{args.feature_mode}_advs.pt")
    test_data = torch.load(f"{feature_dir}/test_{args.feature_mode}_advs.pt")
    data = torch.cat((train_data, test_data), dim=0)

    if args.feature_mode == 'MinAD_KRM':
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4])

    teachermodel=SimpleCNN()
    teacherpath='./../../../modelscls5/parents'
    falseparent,_=get_falsefather_name_and_index(parentname)
    parentlocation=os.path.join(teacherpath,falseparent)
    teachermodel.load_state_dict(torch.load(parentlocation, map_location=args.device))
    teachermodel.to('cuda:0')
    teachermodel.eval()
    
    model['teacher']=teachermodel
    ## 4. IPGuard
    result = {}
    # result[childname]=[]
    for name in model.keys():
        result[name] = []
    # print(len(data),"lendata")
    args.i_num=50
    split_index = random.sample(range(len(data)), args.i_num)
    data_iptest = data[split_index]

    # prediction of teacher model
    target_model = model['teacher']
    target_predict = torch.argmax(target_model(data_iptest), dim=1)
    # prediction of suspected model
    acc=0
    for name in exact_names:
        suspect_model = model[name]
        suspect_model.eval()
        suspect_model.to('cuda:0')
        suspect_predict = torch.argmax(suspect_model(data_iptest), dim=1)
        # compute matching rate
        match_rate = torch.sum(torch.where(target_predict == suspect_predict, 1, 0)) / target_predict.shape[0]
        acc=acc+match_rate
        result[name].append(match_rate.item())
    print(result,"result")
    torch.save(result, './feas/'+parentname)
    totalsubmodel=len(list(result.keys()))-1
    avgacc=acc/totalsubmodel
    print(avgacc,"avgacc")

#####假爷孙关系

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

    parentpath='./../../../modelscls5/parents' 
    parentmodels=os.listdir(parentpath)
    for parentname in parentmodels:                                                                        
        ipguard_function(args,parentname)