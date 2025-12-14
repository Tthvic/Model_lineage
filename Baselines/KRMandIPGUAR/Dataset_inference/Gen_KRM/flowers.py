import os
from PIL import Image
from torch.utils.data import Dataset
# from .load_dataset import load_pickle
import scipy.io
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

class TargetTransform:
    def __init__(self, class_mapping):
        # print(class_mapping,"class_map")
        self.class_mapping = class_mapping

    def __call__(self, target):
        return self.class_mapping[target]


SPLITS = {
    'train': [90, 38, 80, 30, 29, 12, 43, 27, 4, 64, 31, 99, 8, 67, 95, 77,
              78, 61, 88, 74, 55, 32, 21, 13, 79, 70, 51, 69, 14, 60, 11, 39,
              63, 37, 36, 28, 48, 7, 93, 2, 18, 24, 6, 3, 44, 76, 75, 72, 52,
              84, 73, 34, 54, 66, 59, 50, 91, 68, 100, 71, 81, 101, 92, 22,
              33, 87, 1, 49, 20, 25, 58],
    'validation': [10, 16, 17, 23, 26, 47, 53, 56, 57, 62, 82, 83, 86, 97, 102],
    'test': [5, 9, 15, 19, 35, 40, 41, 42, 45, 46, 65, 85, 89, 94, 96, 98],
    'all': list(range(1, 103))
}


class Flowers(Dataset):
    def __init__(self, root, transform=None, target_transform=None,select_classes=[2,3,4,5]):
        self.root = root
        self.transform = transform
        self.select_class=select_classes
        self.target_transform = target_transform
        print(self.target_transform)
        # self.split = 'trainval' if train else 'test'
        self.load_data()

    def _check_exists(self):
        return os.path.exists(self.root)

    def load_data(self,):
        images_path = os.path.join(self.root, 'jpg')
        labels_path = os.path.join(self.root, 'imagelabels.mat')
        labels_mat = scipy.io.loadmat(labels_path)
        image_labels = []
        # split = SPLITS[mode]
        split=self.select_class
        for idx, label in enumerate(labels_mat['labels'][0], start=1):
            if label in split:
                image = str(idx).zfill(5)
                image = f'image_{image}.jpg'
                image = os.path.join(images_path, image)
                # label = split.index(label)
                image_labels.append((image, label))
        # self.xx = image_labels
        data=image_labels
        # data = load_pickle(os.path.join(self.root, 'annotations.pkl'))[self.split]
        # self.samples = [(os.path.join(self.root, 'jpg', i[0]), i[1]) for i in data]
        self.samples = [( i[0], i[1]) for i in data]

    def __getitem__(self, i):
        image, label = self.samples[i]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.samples)


def select_random_classes1(seed):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    num_classes = 2
    select_class = []
    # The original classes that were selected previously
    available_classes = list(set(range(60, 103)))
    for i in range(80):
        # Generate a list of random classes without repetition from the remaining ones
        classes = random.sample(available_classes, num_classes)  # Pick new classes
        select_class.append(classes)

    return select_class


def select_random_classes(seed):
    random.seed(seed)  # 固定随机种子，确保每次选择相同的类别
    num_classes = 6 #######选父模型
    select_class = []
    for i in range(30):
        classes = random.sample(range(0, 60), num_classes)
        select_class.append(classes)
    print(select_class, "select_class")

    return select_class


def get_dataset(selected_class):
    # 数据预处理操作
    # 测试集预处理（与训练集一样）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data_path = './../../../../szy2025/Dataset/Flowers'  # 你的数据集路径
    # 加载完整数据集
    # 初始化 target_transform
    class_mapping = {original_class: new_class for new_class, original_class in enumerate(selected_class)}
    target_transform = TargetTransform(class_mapping)

    dataset = Flowers(root=data_path, transform=transform, target_transform=target_transform,
                   select_classes=selected_class)
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
oriclass=[[40, 7, 1, 47, 17, 15], [14, 8, 47, 6, 43, 57], [57, 34, 5, 37, 27, 2], [1, 5, 13, 14, 32, 38],
 [1, 35, 12, 45, 41, 44], [34, 26, 14, 28, 37, 17], [51, 55, 0, 48, 59, 10], [44, 27, 21, 17, 9, 13],
 [48, 21, 6, 5, 24, 57], [22, 54, 59, 38, 16, 51], [2, 46, 29, 34, 7, 24], [5, 35, 18, 53, 40, 39],
 [56, 55, 23, 36, 12, 45], [4, 2, 42, 14, 49, 18], [5, 54, 14, 55, 6, 24], [17, 29, 40, 53, 23, 10],
 [23, 22, 13, 42, 17, 44], [59, 43, 41, 4, 38, 40], [10, 34, 46, 15, 59, 29], [24, 17, 40, 44, 35, 14],
 [43, 20, 53, 49, 56, 3], [14, 52, 2, 51, 20, 25], [17, 4, 13, 36, 45, 20], [13, 41, 31, 25, 58, 29],
 [9, 16, 8, 15, 47, 35], [34, 16, 47, 37, 27, 56], [25, 23, 14, 8, 32, 31], [5, 48, 3, 55, 7, 9],
 [40, 10, 50, 43, 27, 38], [4, 24, 58, 38, 29, 33]]
ftclass=[
    [96, 65], [91, 76], [62, 60], [69, 102], [97, 90], [83, 80], [61, 77], [91, 72], [86, 94], [94, 66], [72, 96], [
        95, 76], [102, 99], [65, 87], [81, 65], [83, 86], [76, 88], [66, 72], [100, 78], [66, 62], [97, 72], [101,
                                                                                                              83], [
        91, 72], [92, 96], [101, 92], [61, 100], [83, 75], [98, 87], [79, 82], [97, 67], [65, 92], [93, 72], [67,
                                                                                                              98], [
        102, 77], [79, 72], [84, 90], [74, 68], [98, 73], [93, 60], [72, 70], [61, 101], [81, 95], [102, 99], [99,
                                                                                                               79], [
        83, 84], [93, 84], [78, 68], [91, 63], [71, 87], [98, 85], [66, 88], [75, 65], [98, 102], [88, 84], [64,
                                                                                                             93], [
        87, 90], [79, 86], [65, 72], [77, 88], [91, 71], [61, 94], [67, 76], [97, 83], [72, 76], [92, 88], [81,
                                                                                                            93], [
        76, 86], [86, 99], [91, 77], [98, 90], [102, 90], [91, 69], [84, 91], [79, 100], [89, 80], [83, 102], [70,
                                                                                                               99], [
        84, 98], [77, 80], [101, 85]]

if __name__ == "__main__":
    # 获取训练集和测试集
    ###给父模型挑选的类别
    # oriclass=select_random_classes(seed=42)
    # select_random_classes1(seed=43, original_selected_classes=oriclass)
    # ftclass=select_random_classes1(seed=6)

    train_data_loader, test_data_loader = get_data_loader(ftclass[0])
    # # 打印一下加载器中的批次信息（测试数据加载是否正确）
    for idx,(images, labels) in enumerate(train_data_loader):
        print(images.shape,labels,"labels")
    #     print(images.shape, labels.shape)  # 输出每个批次的图像和标签形状
    #     break