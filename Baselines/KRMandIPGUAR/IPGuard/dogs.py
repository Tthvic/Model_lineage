import os
import scipy.io
from os.path import join
import random
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets.utils import download_url, list_dir
import torchvision.transforms as transforms

pclass=[[40, 7, 1, 47, 17, 15, 14, 8], [47, 6, 43, 59, 34, 5, 37, 27], [2, 1, 5, 13, 14, 32, 38, 58],
 [35, 12, 45, 41, 44, 34, 26, 14], [28, 37, 17, 51, 55, 0, 48, 56], [10, 44, 27, 21, 17, 9, 13, 48],
 [21, 6, 5, 24, 58, 22, 54, 38], [16, 51, 2, 46, 29, 34, 7, 24], [5, 35, 18, 53, 40, 39, 23, 36],
 [12, 45, 4, 2, 42, 14, 49, 18], [5, 54, 14, 55, 6, 24, 17, 29], [40, 53, 23, 10, 57, 22, 13, 42],
 [17, 44, 43, 41, 4, 38, 40, 10], [34, 46, 15, 10, 29, 24, 17, 40], [44, 35, 14, 43, 20, 53, 49, 54],
 [3, 14, 52, 2, 51, 20, 25, 17], [4, 13, 36, 56, 45, 20, 58, 41], [31, 25, 56, 41, 29, 9, 16, 8],
 [15, 47, 35, 34, 16, 58, 37, 27], [57, 37, 25, 23, 14, 8, 32, 31]]


childclass=[[84, 81, 74], [70, 117, 118], [84, 96, 117], [104, 78, 95], [77, 84, 114],
            [106, 108, 96], [60, 109, 75], [102, 113, 61], [88, 69, 106], [80, 70, 76],
            [101, 104, 99], [97, 63, 67], [110, 97, 62], [87, 115, 78], [113, 107, 73],
            [64, 106, 83], [106, 90, 68], [95, 102, 61], [67, 66, 118], [69, 104, 117],
            [71, 63, 77], [80, 74, 99], [61, 98, 96], [71, 87, 116], [110, 114, 63],
            [79, 85, 69], [66, 102, 65], [86, 61, 102], [102, 91, 115], [85, 74, 61],
            [65, 68, 64], [89, 106, 85], [87, 94, 107], [77, 94, 61], [116, 60, 101],
            [89, 80, 71], [68, 111, 64], [105, 62, 106], [115, 103, 114], [108, 113, 66],
            [102, 105, 68], [117, 116, 86], [101, 105, 107], [65, 98, 93], [86, 98, 114],
            [66, 75, 74], [87, 83, 106], [89, 103, 106], [86, 110, 88], [65, 102, 77],
            [94, 73, 119], [103, 75, 77], [84, 117, 105], [80, 111, 119], [99, 84, 110],
            [116, 78, 108], [88, 106, 65], [73, 95, 70], [73, 61, 74], [71, 90, 102]]




class Dogs(VisionDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self, root, train, transform=None, target_transform=None, download=False, selected_classes=None):
        super(Dogs, self).__init__(root, transform=transform, target_transform=target_transform)

        self.loader = default_loader
        self.train = train

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]
        print(selected_classes,"selec_class")
        if selected_classes is not None:
            # for image_name, target in self._breed_images:
            #     # print(target,"target")
            #     if target in selected_classes:
            #         print(target,"target")
            self.samples = [(image_name, target) for image_name, target in self._breed_images if target in selected_classes]
        else:
            self.samples = self._breed_images
        # self.samples = self._breed_images

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, target = self.samples[index]
        image_path = join(self.images_folder, image_name)
        image = self.loader(image_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    def stats(self):
        counts = {}
        for index in range(len(self.samples)):
            image_name, target_class = self.samples[index]
            if target_class not in counts.keys():
                counts[target_class] = 1
            else:
                counts[target_class] += 1

        print("%d samples spanning %d classes (avg %f per class)" % (len(self.samples), len(counts.keys()),
                                                                     float(len(self.samples)) / float(
                                                                         len(counts.keys()))))

        return counts
    

class TargetTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, target):
        return self.class_mapping[target]
 
def get_dataset(selected_class):
    # 数据预处理操作
    # 测试集预处理（与训练集一样）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_path = './../../../szy2025/Dataset/Dogs' # 你的数据集路径
    # 加载完整数据集
    # 初始化 target_transform
    class_mapping = {original_class: new_class for new_class, original_class in enumerate(selected_class)}
    target_transform = TargetTransform(class_mapping)
    dataset = Dogs(root=data_path, train=True,transform=transform, target_transform=target_transform, download=False, selected_classes=selected_class)
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


def select_random_classes1(seed):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    num_classes = 3
    select_class = []
    
    # The original classes that were selected previously
    # Flatten the list of all originally selected classes
    # all_original_classes = set([item for sublist in original_selected_classes for item in sublist])

    # all_selected_classes = all_original_classes.copy()  # Track all selected classes, including the original ones
    # available_classes = list(set(range(60, 120)) - all_selected_classes)  # Only consider unselected classes
    available_classes = list(set(range(60, 120)))
    for i in range(60):
        # Generate a list of random classes without repetition from the remaining ones
        classes = random.sample(available_classes, num_classes)  # Pick new classes
        select_class.append(classes)

    return select_class

def select_random_classes(seed):
    # 固定随机种子，确保每次选择相同的类别
    random.seed(seed)
    #######选父模型
    num_classes=8
    select_class=[]
    for i in range(20):
        classes=random.sample(range(0,60), num_classes)
        select_class.append(classes)
    print(select_class,"select_class")

    return  select_class


if __name__ == '__main__':
    data_path='./Dogs'

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # train_dataset = Dogs(data_path, transform=transform, train=True, download=True)
    # val_dataset = Dogs(data_path, transform= transform, train=False, download=True)
    # num_classes = 120
    pclass=select_random_classes(42)
    childclass=select_random_classes1(41)
    print(pclass,"pclass")
    print(childclass,"childclass")
    # for select_class in selected_classes:
    #     print(select_class,"classes")
    #     get_dataset(select_class)