import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import random
import numpy as np

# 定义数据预处理操作
# 定义数据预处理操作（包括调整图像大小为 224x224）
transform = transforms.Compose([
    transforms.Resize((128,128)),  # 调整图像大小为 224x224
    transforms.ToTensor(),          # 转换为 Tensor
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))  # 标准化
])

sunzi_class = [[48, 42], [29, 21], [49, 73], [88, 36], [70, 35], [49, 93], [97, 73], [1, 99], [31, 84], [2, 56],
                [19, 92],
                [40, 21], [32, 83], [88, 79], [74, 7], [15, 75], [4, 55], [36, 94], [27, 9], [92, 46], [93, 60],
                [16, 71], [85, 2],
                [15, 12], [18, 88], [22, 6], [34, 40], [28, 79], [3, 77], [73, 23], [55, 7], [38, 51], [18, 13],
                [85, 11], [52, 2],
                [84, 63], [50, 29], [2, 10], [17, 9], [58, 93], [51, 54], [68, 95], [35, 69], [3, 1], [83, 59],
                [41, 22], [17, 8],
                [91, 5], [92, 87], [96, 13], [84, 90], [16, 52], [83, 90], [95, 10], [77, 67], [53, 76], [12, 30],
                [28, 55],
                [46, 93], [58, 87], [92, 52], [57, 10], [85, 35], [69, 27], [87, 31], [35, 48], [90, 41], [79, 48],
                [37, 96],
                [56, 93], [10, 27], [70, 21], [27, 3], [28, 23], [60, 85], [30, 8], [30, 70], [72, 79], [19, 63],
                [42, 15],
                [29, 49], [86, 92], [15, 57], [76, 94], [91, 24], [70, 99], [63, 50], [45, 34], [45, 3], [22, 75],
                [90, 19],
                [81, 19], [65, 66], [83, 22], [52, 93], [9, 5], [65, 85], [76, 88], [45, 27], [87, 51], [39, 36],
                [19, 33],
                [48, 78], [29, 15], [90, 35], [79, 75], [53, 38], [51, 62], [88, 30], [12, 64], [60, 76], [56, 29],
                [61, 23],
                [60, 37], [9, 31], [63, 60], [72, 48], [6, 78], [91, 90], [68, 3], [96, 97], [80, 25], [52, 63],
                [41, 46],
                [74, 22], [54, 2], [28, 21], [78, 65], [51, 19], [3, 83], [75, 67], [48, 3], [29, 34], [79, 3],
                [71, 98], [67, 93],
                [31, 49], [92, 23], [5, 67], [41, 75], [65, 88], [23, 62], [8, 72], [68, 92], [5, 24], [47, 23],
                [35, 62],
                [11, 96], [84, 56], [2, 91], [39, 13], [38, 49], [8, 69], [40, 85], [56, 96], [91, 47], [65, 22],
                [10, 9],
                [47, 49], [44, 20], [58, 22], [33, 20], [11, 8], [1, 40], [39, 40], [41, 2], [24, 6], [54, 88], [15, 2],
                [19, 92],
                [35, 67], [99, 35], [44, 21], [96, 60], [98, 67], [98, 51], [61, 84], [32, 85], [87, 78], [68, 89],
                [34, 73],
                [88, 20], [31, 3], [96, 35], [66, 6], [24, 85], [41, 36], [36, 32], [37, 34], [36, 11], [8, 33],
                [73, 39],
                [19, 51], [69, 70], [96, 92], [1, 47], [82, 29], [47, 53], [76, 85], [57, 53], [83, 28], [4, 25],
                [7, 37],
                [78, 47], [2, 22], [63, 53], [88, 11], [65, 72], [71, 60], [79, 6], [93, 15], [81, 51], [58, 84],
                [10, 70],
                [93, 88], [11, 97], [89, 98], [2, 11], [87, 72], [35, 57], [55, 49], [34, 4], [19, 79], [79, 82],
                [11, 68],
                [51, 49], [45, 60], [25, 47], [94, 75], [11, 66], [89, 65], [20, 87], [88, 37], [26, 23], [81, 34],
                [34, 62],
                [63, 94], [11, 31], [67, 28], [20, 3], [84, 10], [11, 34], [41, 31], [22, 16], [32, 18], [21, 44],
                [74, 0],
                [63, 10], [10, 9], [71, 90], [14, 45], [82, 18], [11, 25], [35, 46], [31, 9], [18, 38], [10, 20],
                [95, 8],
                [23, 91], [58, 30], [69, 27], [32, 61], [27, 37], [10, 44], [65, 53], [63, 44], [57, 61], [44, 49],
                [79, 97],
                [38, 18], [12, 2], [57, 19], [37, 55], [60, 58], [19, 61], [40, 29], [11, 44], [10, 14], [69, 37],
                [36, 33],
                [30, 42], [27, 0], [65, 81], [79, 67], [59, 95], [85, 92], [75, 54], [8, 76], [89, 64], [59, 29],
                [94, 12],
                [78, 54], [19, 51], [81, 16], [70, 12], [2, 99], [56, 76], [31, 24], [52, 43], [51, 30]]




class TargetTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, target):
        return self.class_mapping[target]


class CIFAR100Subset(Dataset):
    """
    自定义子集类，用于根据指定类别筛选 CIFAR-100 数据。
    """

    def __init__(self, root, train=True, transform=None, selected_classes=None,target_transform=None):
        """
        初始化函数。

        :param root: 数据集存储路径
        :param train: 是否加载训练集
        :param transform: 数据变换
        :param selected_classes: 指定要选择的类别（列表）
        """
        self.dataset = datasets.CIFAR100(root=root, train=train, download=False, transform=transform)
        self.selected_classes = selected_classes
        self.target_transform = target_transform
        # 筛选数据索引
        if selected_classes is not None:
            self.indices = [i for i, label in enumerate(self.dataset.targets) if label in selected_classes]
        else:
            self.indices = list(range(len(self.dataset)))

    def __getitem__(self, index):
        """
        获取指定索引的数据。

        :param index: 索引
        :return: 图像和标签
        """
        real_index = self.indices[index]
        img, label = self.dataset[real_index]
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        """
        返回数据集长度。
        :return: 数据集长度
        """
        return len(self.indices)


def get_data_loader(root='./../../Dataset/CIFAR', batch_size=64, selected_classes=None, train_shuffle=True, test_shuffle=False):
    """
    创建训练集和测试集的 DataLoader，并支持指定类别筛选。

    :param root: 数据集存储路径
    :param batch_size: 批量大小
    :param selected_classes: 指定要选择的类别（列表）
    :param train_shuffle: 是否打乱训练集数据
    :param test_shuffle: 是否打乱测试集数据
    :return: 训练集 DataLoader 和 测试集 DataLoader
    """
    # 创建训练集 DataLoader
    class_mapping = {original_class: new_class for new_class, original_class in enumerate(selected_classes)}
    target_transform = TargetTransform(class_mapping)
    train_dataset = CIFAR100Subset(root=root, train=True, transform=transform, selected_classes=selected_classes,target_transform=target_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, drop_last=False)

    # 创建测试集 DataLoader
    test_dataset = CIFAR100Subset(root=root, train=False, transform=transform, selected_classes=selected_classes,target_transform=target_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=test_shuffle, drop_last=False)

    return train_dataloader, test_dataloader


def select_random_classes(seed):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    #######选父模型
    num_classes = 2
    select_class = []
    for i in range(300):
        classes = random.sample(range(0, 100), num_classes)
        select_class.append(classes)
    print(select_class, "select_class")

    return select_class


# 测试代码
if __name__ == "__main__":
    seed=41
    select_class=select_random_classes(seed)
    print(select_class,"selec_class")
    # 指定要选择的类别（例如：选择类别 0、1、2）
    # select_class = [0, 1, 2]
    #
    # train_loader, test_loader = get_data_loader(root='./../../Dataset/CIFAR', batch_size=64, selected_classes=select_class,
    #                                             train_shuffle=True, test_shuffle=False)
    #
    # # 验证 DataLoader 是否正常工作
    # print("Train DataLoader:")
    # for images, labels in train_loader:
    #     print(labels,"labels")
    #     print(f"Batch size: {images.size(0)}, Image shape: {images.size()}")
    #     break
    #
    # print("Test DataLoader:")
    # for images, labels in test_loader:
    #     print(labels,"labels")
    #     print(f"Batch size: {images.size(0)}, Image shape: {images.size()}")
    #     break
    select_class=[[48, 42], [29, 21], [49, 73], [88, 36], [70, 35], [49, 93], [97, 73], [1, 99], [31, 84], [2, 56], [19, 92],
     [40, 21], [32, 83], [88, 79], [74, 7], [15, 75], [4, 55], [36, 94], [27, 9], [92, 46], [93, 60], [16, 71], [85, 2],
     [15, 12], [18, 88], [22, 6], [34, 40], [28, 79], [3, 77], [73, 23], [55, 7], [38, 51], [18, 13], [85, 11], [52, 2],
     [84, 63], [50, 29], [2, 10], [17, 9], [58, 93], [51, 54], [68, 95], [35, 69], [3, 1], [83, 59], [41, 22], [17, 8],
     [91, 5], [92, 87], [96, 13], [84, 90], [16, 52], [83, 90], [95, 10], [77, 67], [53, 76], [12, 30], [28, 55],
     [46, 93], [58, 87], [92, 52], [57, 10], [85, 35], [69, 27], [87, 31], [35, 48], [90, 41], [79, 48], [37, 96],
     [56, 93], [10, 27], [70, 21], [27, 3], [28, 23], [60, 85], [30, 8], [30, 70], [72, 79], [19, 63], [42, 15],
     [29, 49], [86, 92], [15, 57], [76, 94], [91, 24], [70, 99], [63, 50], [45, 34], [45, 3], [22, 75], [90, 19],
     [81, 19], [65, 66], [83, 22], [52, 93], [9, 5], [65, 85], [76, 88], [45, 27], [87, 51], [39, 36], [19, 33],
     [48, 78], [29, 15], [90, 35], [79, 75], [53, 38], [51, 62], [88, 30], [12, 64], [60, 76], [56, 29], [61, 23],
     [60, 37], [9, 31], [63, 60], [72, 48], [6, 78], [91, 90], [68, 3], [96, 97], [80, 25], [52, 63], [41, 46],
     [74, 22], [54, 2], [28, 21], [78, 65], [51, 19], [3, 83], [75, 67], [48, 3], [29, 34], [79, 3], [71, 98], [67, 93],
     [31, 49], [92, 23], [5, 67], [41, 75], [65, 88], [23, 62], [8, 72], [68, 92], [5, 24], [47, 23], [35, 62],
     [11, 96], [84, 56], [2, 91], [39, 13], [38, 49], [8, 69], [40, 85], [56, 96], [91, 47], [65, 22], [10, 9],
     [47, 49], [44, 20], [58, 22], [33, 20], [11, 8], [1, 40], [39, 40], [41, 2], [24, 6], [54, 88], [15, 2], [19, 92],
     [35, 67], [99, 35], [44, 21], [96, 60], [98, 67], [98, 51], [61, 84], [32, 85], [87, 78], [68, 89], [34, 73],
     [88, 20], [31, 3], [96, 35], [66, 6], [24, 85], [41, 36], [36, 32], [37, 34], [36, 11], [8, 33], [73, 39],
     [19, 51], [69, 70], [96, 92], [1, 47], [82, 29], [47, 53], [76, 85], [57, 53], [83, 28], [4, 25], [7, 37],
     [78, 47], [2, 22], [63, 53], [88, 11], [65, 72], [71, 60], [79, 6], [93, 15], [81, 51], [58, 84], [10, 70],
     [93, 88], [11, 97], [89, 98], [2, 11], [87, 72], [35, 57], [55, 49], [34, 4], [19, 79], [79, 82], [11, 68],
     [51, 49], [45, 60], [25, 47], [94, 75], [11, 66], [89, 65], [20, 87], [88, 37], [26, 23], [81, 34], [34, 62],
     [63, 94], [11, 31], [67, 28], [20, 3], [84, 10], [11, 34], [41, 31], [22, 16], [32, 18], [21, 44], [74, 0],
     [63, 10], [10, 9], [71, 90], [14, 45], [82, 18], [11, 25], [35, 46], [31, 9], [18, 38], [10, 20], [95, 8],
     [23, 91], [58, 30], [69, 27], [32, 61], [27, 37], [10, 44], [65, 53], [63, 44], [57, 61], [44, 49], [79, 97],
     [38, 18], [12, 2], [57, 19], [37, 55], [60, 58], [19, 61], [40, 29], [11, 44], [10, 14], [69, 37], [36, 33],
     [30, 42], [27, 0], [65, 81], [79, 67], [59, 95], [85, 92], [75, 54], [8, 76], [89, 64], [59, 29], [94, 12],
     [78, 54], [19, 51], [81, 16], [70, 12], [2, 99], [56, 76], [31, 24], [52, 43], [51, 30]]
    print(len(select_class),"len_selc")
