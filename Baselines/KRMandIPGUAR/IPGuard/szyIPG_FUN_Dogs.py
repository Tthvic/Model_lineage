import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import torch
import torch.nn as nn
from params import args_parse
from model_data_init import data_init, model_init
from model_train import test_epoch
import re
from utils import *
####szyipguard_function_Calt

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


def get_p_names(file_name):
    # 使用正则表达式匹配父模型名称部分
    import re
    match = re.search(r"Dogs_\d+_(.+?\.pth)\.pth$", file_name)
    if match:
        return match.group(1)  # 返回父模型名称
    else:
        raise ValueError(f"Invalid file name format: {file_name}")


def load_pmodels(pname):
    pmodelpath='./../../../szy2025/train/train_res18/NONIMAGENETPmodels'
    modelpath=os.path.join(pmodelpath,pname)
    newdict=torch.load(modelpath)

    dim1 = 512
    dim2=dimdict['Dogs_p']
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
    dim2 = dimdict['Dogs_c']

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



# python ipguard_function.py --data_name $data_name --feature_mode $feature_mode
# *** main function of IPGuard ***/
def ipguard_function(args,parentname,childnames):

    # suspected models
    # exact_names = ["teacher", "fine-tune", "retrain", "prune", "SA", "DA-LENET", "DA-VGG"]

    ## 1. load the xx
    # train_loader, test_loader = data_init(args.data_name, args.batch_size)

    ## 2. load suspected models
    model = {}
    exact_names=[]
    for name in childnames:
        exactname=parentname+name
        exact_names.append(exactname)
        student=load_cmodels(name)
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
    feature_dir=f"./../Dataset_inference/Gen_KRM/NON_Dogs_adv/"+parentname
    tdata = torch.load(feature_dir)
    data=tdata['adv']
    data=data[0]
    print(data.shape,"data_shape")
    # test_data = torch.load(f"{feature_dir}/test_{args.feature_mode}_advs.pt")
    # data = torch.cat((train_data, test_data), dim=0)

    if args.feature_mode == 'MinAD_KRM':
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2], data.shape[3], data.shape[4])

    teachermodel=load_pmodels(parentname)
    model['teacher']=teachermodel
    ## 4. IPGuard
    result = {}
    # result[childname]=[]
    for name in model.keys():
        result[name] = []
    # print(len(data),"lendata")
    args.i_num=5
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
    return avgacc
#####真父子关系

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

    parentpath='./../../../szy2025/train/train_res18/NONIMAGENETPmodels'
    childpath='./../../../szy2025/train/train_res18/NONCmodels'
    parentmodels=os.listdir(parentpath)
    accs=[]
    for parentname in parentmodels:
        if not 'Dogs' in parentname:
            continue
        childnames=[]
        totchilds=os.listdir(childpath)
        for child in totchilds:
            if 'Dogs' in child:
                currpname= get_p_names(child)
                # if currpname in parentname: 真的样本
                if not currpname in parentname:
                    childnames.append(child)
        acc=ipguard_function(args,parentname,childnames)
        accs.append(acc)
    totalavgaccs=torch.mean(torch.stack(accs))
    print(totalavgaccs,"falsetotalavgacc")