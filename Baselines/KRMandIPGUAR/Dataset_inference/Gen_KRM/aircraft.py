import numpy as np
import os

import random
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, extract_archive
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class TargetTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, target):
        return self.class_mapping[target]

total_class=[[7, 1, 17], [15, 14, 8], [6, 5, 27], [2, 1, 5], [13, 14, 32], [1, 12, 26],
             [14, 28, 17], [0, 10, 27], [21, 17, 9], [13, 21, 6], [5, 24, 6], [22, 16, 2], [29, 7, 24],
             [5, 18, 23], [12, 4, 2], [14, 18, 5], [14, 6, 24], [17, 29, 23], [10, 23, 22], [13, 17, 4],
             [10, 15, 29], [24, 17, 14], [20, 3, 14], [2, 20, 25], [17, 4, 13], [20, 13, 31], [25, 29, 9],
             [16, 8, 15], [16, 27, 25], [23, 14, 8], [32, 31, 5], [3, 7, 9], [10, 27, 4], [24, 29, 33],
             [16, 0, 7], [17, 21, 7], [18, 27, 10], [29, 0, 16], [32, 11, 6], [19, 32, 12], [9, 23, 10],
             [33, 0, 20], [31, 1, 7], [23, 19, 15], [3, 15, 5], [5, 31, 4], [8, 30, 10], [16, 33, 27],
             [13, 12, 19], [25, 23, 28], [33, 28, 7], [15, 14, 4], [21, 1, 14], [14, 0, 4], [3, 14, 4],
             [2, 21, 4], [32, 15, 17], [31, 13, 8], [30, 15, 26], [12, 6, 27], [22, 27, 26], [29, 3, 6],
             [3, 25, 21], [6, 15, 12], [12, 28, 8], [27, 11, 17], [29, 15, 4], [28, 6, 3], [0, 5, 15],
             [10, 26, 31], [30, 13, 25], [3, 10, 24], [0, 24, 16], [29, 18, 27], [31, 9, 12], [18, 13, 3],
             [3, 20, 30], [32, 33, 10], [3, 32, 5], [11, 4, 15], [25, 7, 15], [2, 5, 26], [33, 20, 16],
             [13, 20, 15], [16, 25, 8], [19, 29, 20], [4, 0, 29], [6, 4, 13], [32, 16, 8], [22, 4, 15],
             [23, 18, 10], [28, 19, 33], [0, 19, 6], [8, 16, 7], [6, 9, 17], [18, 13, 21], [13, 16, 32],
             [31, 16, 3], [5, 27, 17], [2, 0, 21], [8, 16, 10], [28, 27, 0], [7, 4, 9], [2, 23, 9],
             [27, 8, 2], [19, 23, 2], [22, 13, 15], [6, 22, 26], [9, 15, 10], [11, 26, 1], [11, 21, 26],
             [15, 17, 10], [6, 24, 2], [30, 14, 12], [29, 22, 19], [14, 1, 12], [25, 21, 17], [4, 17, 22],
             [32, 25, 21], [1, 7, 16], [11, 16, 2], [6, 27, 22], [20, 27, 32], [7, 24, 12], [16, 2, 27],
             [0, 33, 12], [23, 27, 4], [21, 20, 7], [19, 32, 26], [20, 25, 18], [8, 12, 26], [24, 11, 19],
             [25, 0, 19], [18, 13, 27], [20, 29, 28], [28, 13, 32], [30, 10, 5], [18, 32, 21], [5, 15, 19],
             [14, 12, 9], [1, 2, 15], [30, 4, 29], [26, 12, 24], [31, 25, 15], [9, 0, 6], [27, 14, 11],
             [33, 29, 3], [15, 7, 29], [8, 29, 33], [20, 28, 32], [27, 28, 10], [30, 28, 16], [15, 17, 33],
             [31, 15, 17], [28, 4, 18], [15, 17, 21], [20, 5, 8], [9, 14, 24], [9, 13, 4], [26, 21, 29],
             [26, 3, 13], [26, 24, 1], [24, 30, 0], [22, 19, 24], [26, 14, 31], [14, 17, 27], [31, 1, 24],
             [21, 25, 10], [29, 8, 1], [25, 1, 5], [27, 8, 29], [11, 3, 16], [24, 20, 13], [29, 20, 21],
             [24, 17, 26], [16, 5, 30], [1, 3, 22], [14, 4, 2], [1, 15, 12], [1, 9, 15], [8, 30, 7],
             [13, 29, 16], [23, 10, 7], [10, 19, 6], [1, 19, 24], [25, 12, 4], [15, 6, 19], [7, 2, 22],
             [27, 23, 4], [32, 21, 0], [26, 31, 6], [27, 23, 29], [9, 27, 11], [33, 17, 30], [29, 27, 17],
             [20, 15, 5], [17, 28, 15], [29, 24, 21], [1, 31, 20], [11, 31, 13]]

class Aircraft(VisionDataset):
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train, class_type='variant', transform=None,
                 target_transform=None, download=False, selected_classes=None):
        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'

        self.class_type = class_type
        self.split = split
        self.selected_classes = selected_classes  # 需要加载的类别

        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         f'images_{self.class_type}_{self.split}.txt')

        if download:
            self.download()

        image_ids, targets, classes, class_to_idx = self.find_classes()
        self.class_to_idx = class_to_idx
        # print(self.class_to_idx,"class_to_idx")
        # 过滤出属于 selected_classes 的样本
        if self.selected_classes is not None:
            filtered_image_ids = []
            filtered_targets = []
            for img_id, target in zip(image_ids, targets):
                if target in self.selected_classes:
                    filtered_image_ids.append(img_id)
                    filtered_targets.append(target)
            image_ids = filtered_image_ids
            targets = filtered_targets
            # print(image_ids,targets,"ttargets")
        self.samples = self.make_dataset(image_ids, targets)
        self.loader = default_loader
        self.classes = [classes[i] for i in self.selected_classes] if self.selected_classes else classes

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def download(self):
        if self._check_exists():
            return
        print(f'Downloading {self.url}...')
        tar_name = self.url.rpartition('/')[-1]
        download_url(self.url, root=self.root, filename=tar_name)
        tar_path = os.path.join(self.root, tar_name)
        print(f'Extracting {tar_path}...')
        extract_archive(tar_path)
        print('Done!')

    def find_classes(self):
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]
        # print(classes,class_to_idx,targets,"tttargets")
        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert len(image_ids) == len(targets)
        return [(os.path.join(self.root, self.img_folder, f'{img_id}.jpg'), target) for img_id, target in
                zip(image_ids, targets)]


def select_random_classes(listnum, num_classes=8, total_classes=100):
    """ 从前 60 个类别中随机选择 num_classes 个类别 """
    selec_classes=[]
    for i in range(listnum):
        currclass=random.sample(range(total_classes), num_classes)
        selec_classes.append(currclass)
    return  selec_classes


def get_dataloader(root_dir='./../../Dataset/Aircraft', selected_classes=None):
    """ 只加载指定类别的数据 """
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batchsize=128
    class_mapping = {original_class: new_class for new_class, original_class in enumerate(selected_classes)}
    target_transform = TargetTransform(class_mapping)
    train_dataset = Aircraft(root= './../../Dataset/Aircraft', train=True, class_type='variant',
                             transform=train_transform,target_transform=target_transform, download=True, selected_classes=selected_classes)

    test_dataset = Aircraft(root= './../../Dataset/Aircraft', train=False, class_type='variant',
                            transform=test_transform,target_transform=target_transform, download=True, selected_classes=selected_classes)
    print(len(train_dataset),len(test_dataset),"len_traindataset")
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4)

    return train_loader, test_loader


if __name__ == "__main__":
    root_dir = './../../Dataset/Aircraft'
    seed=42
    random.seed(seed)
    # 选择父模型的类别
    listnum=200
    total_classes = select_random_classes(listnum, num_classes=3, total_classes=34)
    print(total_classes,"ppclass")
    for i in range(listnum):
    # 只加载指定类别的数据
        cclass=total_classes[i]
        train_loader, test_loader = get_dataloader(root_dir, cclass)
        # if i<1:
        #     break
