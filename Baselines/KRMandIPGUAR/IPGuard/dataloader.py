import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.folder import default_loader
import random

class FilteredImageFolder(datasets.DatasetFolder):
    def __init__(self, root, class_names, transform=None, loader=default_loader, samples_per_class=None):
        """
        A custom ImageFolder that loads only specified class folders with an optional limit on samples per class.

        Args:
            root (str): Root directory of the dataset.
            class_names (list): List of class names to include.
            transform (callable, optional): A function/transform that takes in an PIL image.
            loader (callable, optional): A function to load an image given its path.
            samples_per_class (int, optional): Number of samples to include per class.
        """
        # Filter classes to include only the specified ones
        all_classes, class_to_idx = self.find_classes(root)
        # filtered_classes = {name: idx for name, idx in class_to_idx.items() if name in class_names}
        filtered_classes = {name: idx for idx, name in enumerate(class_names)}
        self.classes = list(filtered_classes.keys())
        self.class_to_idx = filtered_classes
        print(self.class_to_idx,"self_c")
        # Update root directories to include only filtered classes  
        self.samples = []
        self.img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']  # Valid image extensions

        for class_name in self.classes:
            class_path = os.path.join(root, class_name)
            class_samples = []
            for root_dir, _, file_names in os.walk(class_path):
                for file_name in file_names:
                    if any(file_name.lower().endswith(ext) for ext in self.img_extensions):
                        path = os.path.join(root_dir, file_name)
                        item = (path, self.class_to_idx[class_name])

                        class_samples.append(item)

            # Limit the number of samples per class if specified
            if samples_per_class is not None:
                class_samples = random.sample(class_samples, min(samples_per_class, len(class_samples)))

            self.samples.extend(class_samples)

        self.root = root
        self.loader = loader
        self.extensions = datasets.folder.IMG_EXTENSIONS
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_filtered_tiny_imagenet_dataloaders(data_dir, selected_classes, batch_size=64, num_workers=4, samples_per_class=None, train_split=0.8):
    """
    Create train and test dataloaders for a filtered Tiny ImageNet dataset with specified classes and optional sample limits.

    Args:
        data_dir (str): Path to the Tiny ImageNet data directory.
        selected_classes (list of str): List of class names to include.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for dataloading.
        samples_per_class (int, optional): Number of samples to include per class.
        train_split (float): Proportion of the dataset to use for training.

    Returns:
        tuple: (train_loader, test_loader) DataLoaders for training and testing.
    """
    # Define data transformations
    transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
    ])

    # Create the filtered dataset
    filtered_dataset = FilteredImageFolder(data_dir, selected_classes, transform=transform, samples_per_class=samples_per_class)

    # Split dataset into train and test
    train_size = int(train_split * len(filtered_dataset))
    test_size = len(filtered_dataset) - train_size
    train_dataset, test_dataset = random_split(filtered_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def select_random_classes(data_dir, num_classes=5, num_trials=20):
    """
    从Tiny ImageNet的训练集目录中随机选择`num_classes`个类别，重复`num_trials`次
    """
    # 获取训练集中所有类别的文件夹（每个类别一个子文件夹）
    all_classes = os.listdir(data_dir)
    # 确保目录中只有有效类别文件夹
    all_classes = [cls for cls in all_classes if os.path.isdir(os.path.join(data_dir, cls))]
    
    # 存储每次随机选择的类别
    selected_classes_per_trial = []
    
    # 重复 num_trials 次随机选择
    for _ in range(num_trials):
        selected_classes = random.sample(all_classes, num_classes)
        selected_classes_per_trial.append(selected_classes)
    
    return selected_classes_per_trial

# # 数据集目录
# data_dir = "D:/Datasets/tiny-imagenet-200/train"
# # 随机选择5个类别，重复20次
# random_classes_trials = select_random_classes(data_dir, num_classes=5, num_trials=20)

# # 打印部分结果（每次选择的5个类别）
# for i, classes in enumerate(random_classes_trials, 1):
#     print(classes)

# ['n07734744', 'n02125311', 'n04371430', 'n02226429', 'n01910747']
# ['n01443537', 'n04486054', 'n01784675', 'n03637318', 'n03126707']
# ['n03126707', 'n01784675', 'n02123045', 'n07715103', 'n02233338']
# ['n03085013', 'n03447447', 'n02977058', 'n03977966', 'n02927161']
# ['n04398044', 'n02415577', 'n02906734', 'n04023962', 'n01774750']
# ['n07583066', 'n02410509', 'n04008634', 'n03930313', 'n07579787']
# ['n03160309', 'n07753592', 'n02099601', 'n02509815', 'n02950826']
# ['n03770439', 'n02769748', 'n04596742', 'n02808440', 'n07579787']
# ['n03837869', 'n04597913', 'n03977966', 'n02056570', 'n04501370']
# ['n02279972', 'n01950731', 'n02802426', 'n04146614', 'n04501370']
# ['n09193705', 'n02415577', 'n04560804', 'n03584254', 'n01770393']
# ['n02843684', 'n03970156', 'n03444034', 'n09256479', 'n03649909']
# ['n04146614', 'n03014705', 'n03404251', 'n03617480', 'n04259630']
# ['n04540053', 'n07875152', 'n07768694', 'n02906734', 'n12267677']
# ['n03126707', 'n02233338', 'n04265275', 'n02002724', 'n03983396']
# ['n03255030', 'n03026506', 'n02125311', 'n04376876', 'n04399382']
# ['n04456115', 'n02099601', 'n02268443', 'n04118538', 'n04265275']
# ['n03854065', 'n07695742', 'n04118538', 'n04507155', 'n04067472']
# ['n07875152', 'n03891332', 'n01917289', 'n02480495', 'n04366367']
# ['n02802426', 'n02085620', 'n01774750', 'n02883205', 'n02769748']



# data_dir = "D:/Datasets/tiny-imagenet-200/train"  # Path to Tiny ImageNet training data
# selected_classes = ["n01443537", "n01629819"]  # Specify desired class names
# batch_size = 64
# samples_per_class = 100  # Limit to 100 samples per class

# train_loader, test_loader = get_filtered_tiny_imagenet_dataloaders(
#     data_dir, selected_classes, batch_size=batch_size, samples_per_class=samples_per_class
# )

# print(f"Loaded {len(train_loader.dataset)} training samples and {len(test_loader.dataset)} testing samples from {len(selected_classes)} classes.")
