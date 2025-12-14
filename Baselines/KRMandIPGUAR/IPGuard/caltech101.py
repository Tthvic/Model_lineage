import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import random

oriclass = [[40, 7, 1, 47, 17, 15], [14, 8, 47, 6, 43, 57], [57, 34, 5, 37, 27, 2], [1, 5, 13, 14, 32, 38],
            [1, 35, 12, 45, 41, 44], [34, 26, 14, 28, 37, 17], [51, 55, 0, 48, 59, 10], [44, 27, 21, 17, 9, 13],
            [48, 21, 6, 5, 24, 57], [22, 54, 59, 38, 16, 51], [2, 46, 29, 34, 7, 24], [5, 35, 18, 53, 40, 39],
            [56, 55, 23, 36, 12, 45], [4, 2, 42, 14, 49, 18], [5, 54, 14, 55, 6, 24], [17, 29, 40, 53, 23, 10],
            [23, 22, 13, 42, 17, 44], [59, 43, 41, 4, 38, 40], [10, 34, 46, 15, 59, 29], [24, 17, 40, 44, 35, 14]]
ftclass = [[93, 62, 88, 73], [58, 50, 66, 99], [94, 87, 80, 77], [52, 74, 88, 69], [83, 91, 100, 63], [69, 93, 92, 73],
           [99, 96, 100, 62], [84, 78, 62, 80], [83, 73, 85, 63], [69, 97, 75, 63], [58, 94, 69, 98], [80, 88, 69, 89],
           [93, 98, 89, 52], [97, 80, 72, 95], [84, 76, 79, 94], [64, 62, 89, 100], [90, 69, 64, 95], [99, 74, 76, 69],
           [81, 87, 71, 65], [95, 70, 90, 50], [69, 67, 52, 98], [78, 92, 99, 96], [96, 76, 80, 81], [90, 81, 75, 65],
           [100, 88, 60, 68], [84, 95, 82, 63], [85, 72, 62, 95], [99, 85, 81, 61], [90, 84, 87, 76], [83, 62, 69, 100],
           [74, 85, 88, 68], [52, 91, 64, 73], [94, 80, 69, 73], [89, 85, 78, 90], [73, 83, 96, 88], [74, 95, 87, 99],
           [87, 88, 66, 81], [88, 76, 97, 86], [77, 80, 99, 67], [96, 81, 95, 74], [77, 98, 82, 88], [67, 75, 92, 50],
           [96, 85, 60, 68], [52, 96, 93, 64], [100, 80, 88, 94], [60, 69, 66, 74], [96, 50, 84, 90], [88, 61, 87, 71],
           [63, 80, 66, 100], [97, 72, 95, 77], [66, 58, 97, 99], [63, 58, 98, 87], [86, 61, 97, 52], [92, 65, 96, 62],
           [67, 96, 97, 73], [98, 85, 88, 52], [97, 65, 69, 85], [87, 85, 92, 74], [87, 94, 61, 75], [80, 75, 58, 62]]


class TargetTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, target):
        return self.class_mapping[target]


class Caltech101(Dataset):
    def __init__(self, root, selected_classes=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.selected_classes = selected_classes  # List of class indices to include
        self.transform = transform
        self.target_transform = target_transform
        self.split = 'train' if train else 'test'
        self.data = self.load_data()

    def load_data(self):
        data_folder = os.path.join(self.root, '101_ObjectCategories')
        dataset = ImageFolder(root=data_folder, transform=self.transform)
        
        if self.selected_classes is not None:
            # Filter data based on selected classes
            selected_data = []
            tmp=0
            for item in dataset:
                # if tmp>2:
                #     continue
                # print(item)
                if item[1] in self.selected_classes:
                    selected_data.append(item)
                tmp=tmp+1
            print(len(selected_data), "len_selc_data")
            return selected_data
        else:
            print(self.split, "self_split")
            # If no selected_classes, return all data
            if self.split == 'train':
                return [item for item in dataset if item[1] < 50]  # First 50 categories as training set
            else:
                return [item for item in dataset if item[1] >= 50]  # Last 50 categories as test set

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # if self.transform:
        #     # image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)

    
def get_dataset(selected_class):
    # 数据预处理操作
    # 测试集预处理（与训练集一样）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_path = './../../../szy2025/Dataset/Caltech101/caltech-101'  # 你的数据集路径
    # 加载完整数据集
    # 初始化 target_transform
    class_mapping = {original_class: new_class for new_class, original_class in enumerate(selected_class)}
    target_transform = TargetTransform(class_mapping)
    dataset = Caltech101(data_path, selected_classes=selected_class, train=True, transform=transform, target_transform=target_transform)
    
    # 使用random_split来将数据集拆分为训练集和测试集
    train_size = int(0.8 * len(dataset))  # 80%作为训练集
    test_size = len(dataset) - train_size  # 剩下的20%作为测试集
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def get_data_loader(select_class):
    train_dataset, test_dataset = get_dataset(select_class)
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)
    return train_data_loader, test_data_loader



def select_random_classes1(seed,original_selected_classes):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    num_classes = 4
    select_class = []
    # Flatten the list of all originally selected classes
    all_original_classes = set([item for sublist in original_selected_classes for item in sublist])
    
    all_selected_classes = all_original_classes.copy()  # Track all selected classes, including the original ones
    available_classes = list(set(range(50,101)) - all_selected_classes)  # Only consider unselected classes
    print(len(available_classes),"ava_class")
    for i in range(60):
        # Generate a list of random classes without repetition from the remaining ones
        print(all_original_classes,"allorig",len(all_selected_classes))
        classes = random.sample(available_classes, num_classes)  # Pick new classes
        select_class.append(classes)
        all_selected_classes.update(classes)  # Add the selected classes to the set of selected classes

    print(select_class, "select_class")
    return select_class

def select_random_classes(seed):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    #######选父模型
    num_classes=6
    select_class=[]
    for i in range(20):
        classes=random.sample(range(0, 60), num_classes)
        select_class.append(classes)
    print(select_class,"select_class")
    # select_classes = [3, 44, 55, 22, 1]
    # 动态生成 class_mapping 字典
    # class_mapping = {original_class: new_class for new_class, original_class in enumerate(select_classes)}
    # 初始化 target_transform
    # target_transform = TargetTransform(class_mapping)
    # [[81, 14, 3, 94, 35, 31, 28, 17], [94, 13, 86, 69, 11, 75, 54, 4], [3, 11, 27, 29, 64, 77, 71, 25],
    #   [91, 83, 89, 69, 53, 28, 57, 75], [35, 0, 97, 20, 89, 54, 43, 19], [27, 97, 43, 13, 11, 48, 12, 45]]
    return   select_class

chongsunclass=[[25, 12], [21, 57], [11, 40], [0, 33], [49, 44], [0, 54], [21, 43], [29, 8],
            [14, 7], [44, 8], [38, 11], [28, 26], [20, 42], [27, 8], [19, 10], [7, 29],
            [52, 55], [21, 3], [9, 57], [15, 28], [56, 12], [14, 11], [58, 0], [15, 41],
            [33, 29], [50, 33], [1, 52], [31, 45], [24, 32], [24, 30], [32, 4], [31, 9],
            [24, 0], [8, 27], [56, 19], [45, 54], [41, 1], [56, 8], [6, 35], [33, 5],
            [24, 18], [21, 45], [13, 27], [23, 32], [1, 54], [29, 45], [22, 36], [20, 38],
            [39, 38], [24, 10], [22, 19], [43, 27], [33, 48], [36, 50], [21, 26], [30, 2],
            [40, 19], [27, 51], [14, 59], [26, 52], [22, 2], [3, 20], [15, 32], [28, 44],
            [49, 50], [45, 18], [5, 35], [3, 47], [19, 9], [2, 22], [35, 57], [12, 59],
            [48, 42], [14, 47], [1, 27], [30, 58], [8, 40], [24, 2], [46, 32], [43, 44],
            [8, 47], [49, 57], [30, 19], [43, 39], [39, 18], [5, 56], [44, 18], [33, 56],
            [36, 58], [28, 12], [17, 47], [0, 2], [55, 20], [10, 12], [44, 24], [58, 38],
            [48, 11], [30, 33], [47, 25], [36, 44]]


def select_random_classes_forchongsun(seed):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    #######选父模型
    num_classes=2
    select_class=[]
    for i in range(100):
        classes=random.sample(range(0, 60), num_classes)
        select_class.append(classes)
    return   select_class



if __name__ == "__main__":
    # 获取训练集和测试集
    ###给父模型挑选的类别
    # oriclass=select_random_classes(seed=42)
    # # select_random_classes1(seed=43, original_selected_classes=oriclass)
    # ftclass=select_random_classes1(seed=6, original_selected_classes=oriclass)
    # print(oriclass,"oriclass")
    # print(ftclass,"ftclass")
    # oriclass=[[40, 7, 1, 47, 17, 15], [14, 8, 47, 6, 43, 57], [57, 34, 5, 37, 27, 2], [1, 5, 13, 14, 32, 38],
    #  [1, 35, 12, 45, 41, 44], [34, 26, 14, 28, 37, 17], [51, 55, 0, 48, 59, 10], [44, 27, 21, 17, 9, 13],
    #  [48, 21, 6, 5, 24, 57], [22, 54, 59, 38, 16, 51], [2, 46, 29, 34, 7, 24], [5, 35, 18, 53, 40, 39],
    #  [56, 55, 23, 36, 12, 45], [4, 2, 42, 14, 49, 18], [5, 54, 14, 55, 6, 24], [17, 29, 40, 53, 23, 10],
    #  [23, 22, 13, 42, 17, 44], [59, 43, 41, 4, 38, 40], [10, 34, 46, 15, 59, 29], [24, 17, 40, 44, 35, 14]]
    # ftclass=[[93, 62, 88, 73], [58, 50, 66, 99], [94, 87, 80, 77], [52, 74, 88, 69], [83, 91, 100, 63], [69, 93, 92, 73],
    #  [99, 96, 100, 62], [84, 78, 62, 80], [83, 73, 85, 63], [69, 97, 75, 63], [58, 94, 69, 98], [80, 88, 69, 89],
    #  [93, 98, 89, 52], [97, 80, 72, 95], [84, 76, 79, 94], [64, 62, 89, 100], [90, 69, 64, 95], [99, 74, 76, 69],
    #  [81, 87, 71, 65], [95, 70, 90, 50], [69, 67, 52, 98], [78, 92, 99, 96], [96, 76, 80, 81], [90, 81, 75, 65],
    #  [100, 88, 60, 68], [84, 95, 82, 63], [85, 72, 62, 95], [99, 85, 81, 61], [90, 84, 87, 76], [83, 62, 69, 100],
    #  [74, 85, 88, 68], [52, 91, 64, 73], [94, 80, 69, 73], [89, 85, 78, 90], [73, 83, 96, 88], [74, 95, 87, 99],
    #  [87, 88, 66, 81], [88, 76, 97, 86], [77, 80, 99, 67], [96, 81, 95, 74], [77, 98, 82, 88], [67, 75, 92, 50],
    #  [96, 85, 60, 68], [52, 96, 93, 64], [100, 80, 88, 94], [60, 69, 66, 74], [96, 50, 84, 90], [88, 61, 87, 71],
    #  [63, 80, 66, 100], [97, 72, 95, 77], [66, 58, 97, 99], [63, 58, 98, 87], [86, 61, 97, 52], [92, 65, 96, 62],
    #  [67, 96, 97, 73], [98, 85, 88, 52], [97, 65, 69, 85], [87, 85, 92, 74], [87, 94, 61, 75], [80, 75, 58, 62]]
    finclass=select_random_classes_forchongsun(seed=88)
    print(finclass,"finclass")
    # 获取数据加载器
    ####给子模型挑选的类别
    # select_random_classes1(seed=41)
    # select_classes=[[81, 14, 3, 94, 35, 31, 28, 17], [94, 13, 86, 69, 11, 75, 54, 4], [3, 11, 27, 29, 64, 77, 71, 25],
    #    [91, 83, 89, 69, 53, 28, 57, 75], [35, 0, 97, 20, 89, 54, 43, 19], [27, 97, 43, 13, 11, 48, 12, 45]]

    # train_data_loader, test_data_loader = get_data_loader(train_dataset, test_dataset)
    # # 打印一下加载器中的批次信息（测试数据加载是否正确）
    # for images, labels in train_data_loader:
    #     print(labels,"labels")
    #     print(images.shape, labels.shape)  # 输出每个批次的图像和标签形状
    #     break