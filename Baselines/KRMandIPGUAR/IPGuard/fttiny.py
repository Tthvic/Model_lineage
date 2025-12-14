import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
# from src.task_vectors import TaskVector
import torchvision
import random 
import numpy as np
import os
from traincifar import SimpleCNN
from dataloader import get_filtered_tiny_imagenet_dataloaders

def seed_everything(seed):
    """
    固定所有可能的随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 设置随机种子
# seed_everything(42)

# 定义超参数
def main():
     # 定义模型结构
    # 测试模型
    def test_model(model,test_loader,criterion):
        model.eval()
        model=model.to('cuda:0')
        test_loss = 0
        correct = 0
        totalsamples=len(test_loader.dataset)
        with torch.no_grad():
            kkk=0
            for data, target in test_loader:
                if kkk>1:
                    continue
                data=data.to('cuda:0')
                target=target.to('cuda:0')
                # print(target,"target")
                output = model(data)
                # target=target%4
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                kkk=kkk+1
        test_loss /= len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy:({100. * correct /target.shape[0]:.0f}%)\n')

    def trainftmodel(trainloader,testloader,father,num_epochs=5,modelname='newszy'):
        learning_rate = 0.001
        model=father.to('cuda:0')
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer1 = optim.Adam(model.parameters(), lr=learning_rate)
        # train_dataloader,test_dataloader=get_dataloaders(selected_classes,batch_size) 
        for epoch in range(num_epochs): 
            model.train()  # 设置模型为训练模式
            correct = 0  # 初始化正确预测数
            total = 0    # 初始化总样本数
            for batch_idx, (data, target) in enumerate(trainloader):
                data = data.to('cuda:0')
                target = target.to('cuda:0')
                optimizer1.zero_grad()  # 清零梯度
                output = model(data)  # 前向传播
                # 计算损失
                loss = criterion(output, target)
                loss.backward()  # 反向传播
                optimizer1.step()  # 更新参数
                # 计算正确预测的数量
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率对应的类别
                correct += pred.eq(target.view_as(pred)).sum().item()  # 更新正确预测数
                total += target.size(0)  # 更新总样本数
            # 计算并打印准确率
            accuracy = 100. * correct / total
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
            test_model(model,testloader,criterion)
        torch.save(model.state_dict(),modelname)

        return model

    # Example usage
    seed_everything(555) #####选取 tinyimagenet进行微调 
    totalclass=[] 
   
    #####10分类
    # totalselected_classes=[
    #     ['n02403003', 'n02814533', 'n02233338', 'n02124075', 'n04118538', 'n07695742', 'n02099712', 'n02226429', 'n02791270', 'n03404251'],
    #     ['n02403003', 'n04070727', 'n07715103', 'n07875152', 'n02815834', 'n04285008', 'n03662601', 'n01698640', 'n02124075', 'n04254777'],
    #     ['n01774384', 'n03837869', 'n03637318', 'n01784675', 'n02099712', 'n04562935', 'n02123394', 'n02056570', 'n02948072', 'n04275548'],
    #     ['n04275548', 'n07747607', 'n03424325', 'n02791270', 'n02927161', 'n02190166', 'n07715103', 'n04417672', 'n02123394', 'n07920052'],
    #     ['n04008634', 'n06596364', 'n04465501', 'n01944390', 'n04099969', 'n04501370', 'n02504458', 'n02814533', 'n02823428', 'n02917067'],
    #     ['n02231487', 'n07734744', 'n02165456', 'n07695742', 'n02883205', 'n09193705', 'n01950731', 'n02769748', 'n01984695', 'n04251144'],
    #     ['n02403003', 'n02837789', 'n04597913', 'n09332890', 'n07715103', 'n02883205', 'n02802426', 'n03404251', 'n02486410', 'n02113799'],
    #     ['n02814533', 'n02808440', 'n03804744', 'n03992509', 'n02437312', 'n01983481', 'n04540053', 'n02927161', 'n04596742', 'n04285008'],
    #     ['n09332890', 'n03976657', 'n02814860', 'n03544143', 'n03400231', 'n01629819', 'n04328186', 'n07734744', 'n03891332', 'n06596364'],
    #     ['n03891332', 'n01882714', 'n04465501', 'n04501370', 'n03992509', 'n04265275', 'n02403003', 'n02814860', 'n09193705', 'n07614500'],
    #     ['n03042490', 'n04560804', 'n01917289', 'n03201208', 'n03584254', 'n02364673', 'n07753592', 'n07749582', 'n02481823', 'n02883205'],
    #     ['n02415577', 'n02099712', 'n04532106', 'n03201208', 'n04328186', 'n02699494', 'n02963159', 'n02002724', 'n04133789', 'n04311004'],
    #     ['n03891332', 'n02124075', 'n02795169', 'n07615774', 'n02233338', 'n03838899', 'n07747607', 'n01443537', 'n01644900', 'n02113799'],
    #     ['n07720875', 'n04540053', 'n02125311', 'n03100240', 'n02403003', 'n02233338', 'n07734744', 'n03649909', 'n01629819', 'n02206856'],
    #     ['n02058221', 'n04532106', 'n04532670', 'n02795169', 'n04596742', 'n02281406', 'n03796401', 'n02125311', 'n03444034', 'n02124075'],
    #     ['n03444034', 'n03891332', 'n04562935', 'n07753592', 'n02395406', 'n07695742', 'n01983481', 'n03100240', 'n04328186', 'n03980874'],
    #     ['n02056570', 'n02058221', 'n03706229', 'n04507155', 'n02814860', 'n04275548', 'n04456115', 'n02190166', 'n01774384', 'n03544143'],
    #     ['n07711569', 'n04118538', 'n02423022', 'n07734744', 'n09428293', 'n07715103', 'n03770439', 'n04275548', 'n07920052', 'n03388043'],
    #     ['n01882714', 'n03814639', 'n02056570', 'n02808440', 'n03649909', 'n03388043', 'n02226429', 'n04285008', 'n02814533', 'n09256479'],
    #     ['n02909870', 'n04149813', 'n01770393', 'n03617480', 'n02074367', 'n04070727', 'n03983396', 'n03763968', 'n02841315', 'n03649909']  ]   
    #######5分类
    totalselected_classes=[['n07734744', 'n02125311', 'n04371430', 'n02226429', 'n01910747'],
    ['n01443537', 'n04486054', 'n01784675', 'n03637318', 'n03126707'],
    ['n03126707', 'n01784675', 'n02123045', 'n07715103', 'n02233338'],
    ['n03085013', 'n03447447', 'n02977058', 'n03977966', 'n02927161'],
    ['n04398044', 'n02415577', 'n02906734', 'n04023962', 'n01774750'],
    ['n07583066', 'n02410509', 'n04008634', 'n03930313', 'n07579787'],
    ['n03160309', 'n07753592', 'n02099601', 'n02509815', 'n02950826'],
    ['n03770439', 'n02769748', 'n04596742', 'n02808440', 'n07579787'],
    ['n03837869', 'n04597913', 'n03977966', 'n02056570', 'n04501370'],
    ['n02279972', 'n01950731', 'n02802426', 'n04146614', 'n04501370'],
    ['n09193705', 'n02415577', 'n04560804', 'n03584254', 'n01770393'],
    ['n02843684', 'n03970156', 'n03444034', 'n09256479', 'n03649909'],
    ['n04146614', 'n03014705', 'n03404251', 'n03617480', 'n04259630'],
    ['n04540053', 'n07875152', 'n07768694', 'n02906734', 'n12267677'],
    ['n03126707', 'n02233338', 'n04265275', 'n02002724', 'n03983396'],
    ['n03255030', 'n03026506', 'n02125311', 'n04376876', 'n04399382'],
    ['n04456115', 'n02099601', 'n02268443', 'n04118538', 'n04265275'],
    ['n03854065', 'n07695742', 'n04118538', 'n04507155', 'n04067472'],
    ['n07875152', 'n03891332', 'n01917289', 'n02480495', 'n04366367'],
    ['n02802426', 'n02085620', 'n01774750', 'n02883205', 'n02769748']]

    parentmodelnames=os.listdir('./../modelscls5/parents')
    batch_size=500
    data_dir= "D:/Datasets/tiny-imagenet-200/train"
    for j in range(10):
        selected_classes=totalselected_classes[j]
        trainloader,testloader=get_filtered_tiny_imagenet_dataloaders(
                data_dir, selected_classes, batch_size=batch_size, samples_per_class=100)
        for modelname in parentmodelnames:
            if os.path.exists(os.path.join('./../modelscls5/childs',modelname+'dataset'+str(j)+'.pth')):
                continue
            # if 'seed666' in modelname:        
            #     continue              
            father=SimpleCNN()
            father.load_state_dict(torch.load(os.path.join('./../modelscls5/parents',modelname) ))
            father.to('cuda:0')
            childname=os.path.join('./../modelscls5/childs',modelname+'dataset'+str(j)+'.pth')
            trainftmodel(trainloader,testloader,father,num_epochs=10,modelname=childname)


if __name__ == "__main__":
    main()
 