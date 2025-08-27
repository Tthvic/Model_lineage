import os
import torch
import re
from torch.utils.data import Dataset,TensorDataset, DataLoader, random_split, Subset

pathdict={
    'Calt_sun':'SUN_CIFAR_Calt_adv_fea',
    'Dogs_sun':'SUN_CIFAR_Dogs_adv_fea',
    'Flowers_sun': 'SUN_CIFAR_Flowers_adv_fea',
    'Calt_chongsun': 'CSUN_Aircraft_CIFAR_Calt_adv_fea',
    'Dogs_chongsun': 'CSUN_Pet_CIFAR_Dogs_adv_fea',
    'Flowers_chongsun': 'CSUN_Calt_CIFAR_Flowers_adv_fea',
    'Calt_parent_child':'Calt_adv_fea',
     'Dogs_parent_child':'Dogs_adv_fea',
    'Flowers_parent_child':'Flowers_adv_fea'
}


def simplify_filename(file_name):
    """
    Simplifies the file name by removing the extra '.pth' part using string operations.

    :param file_name: The input file name to process.
    :return: The simplified file name.
    """
    # Split the file name into parts
    parts = file_name.split(".")

    if len(parts) >= 3 and parts[-2] == "pth":  # Check if there are redundant 'pth' parts
        # Reconstruct the simplified file name
        simplified_name = f"{'.'.join(parts[:-2])}.{parts[-1]}"
        return simplified_name
    else:
        return file_name  # Return the original name if no simplification is needed


# L2 normalization helper
# def l2_normalize(tensor):
#     norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
#     return tensor / (norm + 1e-8)

def l2_normalize(tensor, min_val=None, max_val=None):
    """
    Perform Min-Max normalization on the last dimension of a tensor.

    :param tensor: Input tensor of shape (..., d).
    :param min_val: Minimum value for normalization (optional).
    :param max_val: Maximum value for normalization (optional).
    :return: Min-Max normalized tensor.
    # """
    # if min_val is None:
    #     min_val = tensor.min(dim=-1, keepdim=True)[0]
    # if max_val is None:
    #     max_val = tensor.max(dim=-1, keepdim=True)[0]
    # return (tensor - min_val) / (max_val - min_val + 1e-8)
    if torch.isnan(tensor).any():  # check for NaNs in weights
        print(f"Detected NaN in parameter:BAFEA")
    return  tensor


def get_p_names(file_name):
    # extract parent model name using regex
    match = re.search(r"Calt_\d+_(.+?)$", file_name)
    if match:
        return match.group(1)  # 返回父模型名称
    else:
        raise ValueError(f"Invalid file name format: {file_name}")


 # Check and pad features to [64, 512] if necessary
# padding helpers to shapes (N, 512), (8,8,512) or (6,8,512)
def pad_to_64_512(tensor):
    if tensor.shape[0] == 64:
        return tensor.reshape(8, 8, 512)
    elif tensor.shape[0] == 36:
        padded_tensor = torch.zeros((6, 8, 512), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, :6, :] = tensor.reshape(6, 6, 512)
        return padded_tensor
    else:
        raise ValueError("Unsupported tensor shape for padding")



class EmbeddingDataset(Dataset):
    def __init__(self, ):
        self.datapath='./../train/Gen_feas_res18'
        self.dataset,self.falsedataset = self.load_data_total()

    def __len__(self):
        """Returns the total number of models."""
        return len(self.dataset)

    def load_data_total(self,):
        dataset=[]
        falsedataset=[]
        # Calt_fea_path='Calt_adv_fea'
        # dataset,falsedataset=self.load_data(Calt_fea_path,dataset,falsedataset)
        # Dogs_fea_path='Dogs_adv_fea'
        # dataset,falsedataset=self.load_data(Dogs_fea_path,dataset,falsedataset)
        # Flowers_fea_path='Flowers_adv_fea'
        # dataset,falsedataset=self.load_data(Flowers_fea_path,dataset,falsedataset)
        nonCalt_fea_path='non_Calt_adv_fea'
        dataset,falsedataset=self.load_data(nonCalt_fea_path,dataset,falsedataset)
        nonDogs_fea_path='non_Dogs_adv_fea'
        dataset,falsedataset=self.load_data(nonDogs_fea_path,dataset,falsedataset)
        nonFlowers_fea_path='non_Flowers_adv_fea'
        dataset,falsedataset=self.load_data(nonFlowers_fea_path,dataset,falsedataset)

        return dataset,falsedataset


    def load_data(self,eachdatapath,dataset,falsedataset):
        datapath = os.path.join(self.datapath,eachdatapath)
        for cname in os.listdir(datapath):  ##### order: b_a, child, parent
            cdata=torch.load(os.path.join(datapath,cname))
            b_afea=l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea=l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea=l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea=l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea=l2_normalize(cdata['fcfea'].squeeze().squeeze())
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)
            for i in range(b_afea.shape[0]):
                dataset.append((cname,[b_afea[i],c_fea[i],p_fea[i]] ))
                falsedataset.append((cname,[fb_afea[i],fcfea[i],p_fea[i]]))
            if torch.isnan(b_afea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:BAFEA")
            if torch.isnan(p_fea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:PFEA")
            if torch.isnan(c_fea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:C")
            if torch.isnan(fb_afea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:FBA")

        return dataset,falsedataset


    def get_query_grandparent_child(self,grandparent_childpath):
        # grandparent_childpath='SUN_CIFAR_Calt_adv_fea'
        datapath = os.path.join(self.datapath,grandparent_childpath)
        grandparent_childdata=[]
        falsegrandparent_childdata=[]
        for cname in os.listdir(datapath):  ##### order: b_a, child, parent
            cdata=torch.load(os.path.join(datapath,cname))
            b_afea=l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea=l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea=l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea=l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea=l2_normalize(cdata['fcfea'].squeeze().squeeze())
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)
            for i in range(b_afea.shape[0]):
                grandparent_childdata.append((cname,[b_afea[i],c_fea[i],p_fea[i]] ))
                falsegrandparent_childdata.append((cname,[fb_afea[i],fcfea[i],p_fea[i]]))
            if torch.isnan(b_afea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:BAFEA")
            if torch.isnan(p_fea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:PFEA")
            if torch.isnan(c_fea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:C")
            if torch.isnan(fb_afea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:FBA")
            # convert to tensors
        grandparent_childdata = torch.stack([torch.stack(sample[1]) for sample in grandparent_childdata])
        falsegrandparent_childdata = torch.stack([torch.stack(sample[1]) for sample in falsegrandparent_childdata])
        grandparent_childdata=grandparent_childdata.transpose(0, 1)
        falsegrandparent_childdata=falsegrandparent_childdata.transpose(0, 1)
        return grandparent_childdata,falsegrandparent_childdata

    def get_query_chongsun(self,chongsunpath):
        # chongsunpath='CSUN_Aircraft_CIFAR_Calt_adv_fea'
        datapath = os.path.join(self.datapath,chongsunpath)
        chongsundata=[]
        falsechongsundata=[]
        for cname in os.listdir(datapath):  ##### order: b_a, child, parent
            cdata=torch.load(os.path.join(datapath,cname))
            b_afea=l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea=l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea=l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea=l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea=l2_normalize(cdata['fcfea'].squeeze().squeeze())
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)
            for i in range(b_afea.shape[0]):
                chongsundata.append((cname,[b_afea[i],c_fea[i],p_fea[i]] ))
                falsechongsundata.append((cname,[fb_afea[i],fcfea[i],p_fea[i]]))
            if torch.isnan(b_afea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:BAFEA")
            if torch.isnan(p_fea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:PFEA")
            if torch.isnan(c_fea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:C")
            if torch.isnan(fb_afea).any():  # check NaN in weights
                print(f"Detected NaN in parameter:FBA")
            # 转换为张量
        # grandparent_childdata = torch.stack([torch.stack(sample[1]) for sample in grandparent_childdata])
        chongsundata = torch.stack([torch.stack(sample[1]) for sample in chongsundata])
        falsechongsundata = torch.stack([torch.stack(sample[1]) for sample in falsechongsundata])
        chongsundata=chongsundata.transpose(0, 1)
        falsechongsundata=falsechongsundata.transpose(0, 1)
        return chongsundata,falsechongsundata

    def get_query_non(self):
        dataset=[]
        falsedataset=[]
        nonCalt_fea_path='Calt_adv_fea'
        dataset,falsedataset=self.load_data(nonCalt_fea_path,dataset,falsedataset)
        nonDogs_fea_path='Dogs_adv_fea'
        dataset,falsedataset=self.load_data(nonDogs_fea_path,dataset,falsedataset)
        nonFlowers_fea_path='Flowers_adv_fea'
        dataset,falsedataset=self.load_data(nonFlowers_fea_path,dataset,falsedataset)
        # nonCalt_fea_path='non_Calt_adv_fea'
        # dataset,falsedataset=self.load_data(nonCalt_fea_path,dataset,falsedataset)
        # nonDogs_fea_path='non_Dogs_adv_fea'
        # dataset,falsedataset=self.load_data(nonDogs_fea_path,dataset,falsedataset)
        # nonFlowers_fea_path='non_Flowers_adv_fea'
        # dataset,falsedataset=self.load_data(nonFlowers_fea_path,dataset,falsedataset)
        dataset = torch.stack([torch.stack(sample) for sample in dataset[1]])
        falsedataset = torch.stack([torch.stack(sample) for sample in falsedataset[1]])
        dataset=dataset.transpose(0, 1)
        falsedataset=falsedataset.transpose(0, 1)
        return  dataset,falsedataset

    def get_query_parent_child(self,datapath):
        dataset = []
        falsedataset = []
        dataset, falsedataset = self.load_data(datapath, dataset, falsedataset)
        dataset = torch.stack([torch.stack(sample[1]) for sample in dataset])
        falsedataset = torch.stack([torch.stack(sample[1]) for sample in falsedataset])
        dataset = dataset.transpose(0, 1)
        falsedataset = falsedataset.transpose(0, 1)
        return dataset, falsedataset

    def get_query_distall(self,):
        def padd(input_tensor):
            # average pool across spatial dims 8x8
            pooled = torch.mean(input_tensor, dim=(2, 3))
            # concatenate 4 times across channel to 512 dims (128 * 4 = 512)
            output = torch.cat([pooled, pooled, pooled, pooled], dim=1)
            return output

        datapath='./../Distil/genadv/Calt_adv_fea'
        falsedataset=[]
        dataset=[]
        # datapath = os.path.join(self.datapath,eachdatapath)
        for cname in os.listdir(datapath):  ##### order: b_a, child, parent
            cdata=torch.load(os.path.join(datapath,cname))
            b_afea=l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea=l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea=l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea=l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea=l2_normalize(cdata['fcfea'].squeeze().squeeze())
            # print(b_afea.shape,"shape")
            b_afea,p_fea,c_fea=padd(b_afea),padd(p_fea),padd(c_fea)
            fb_afea,fcfea=padd(fb_afea),padd(fcfea)        
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)
            for i in range(b_afea.shape[0]):
                dataset.append([b_afea[i],c_fea[i],p_fea[i]] )
                falsedataset.append([fb_afea[i],fcfea[i],p_fea[i]])
            if torch.isnan(b_afea).any(): 
                print(f"Detected NaN in parameter:BAFEA")
            if torch.isnan(p_fea).any():
                print(f"Detected NaN in parameter:PFEA")
            if torch.isnan(c_fea).any():
                print(f"Detected NaN in parameter:C")
            if torch.isnan(fb_afea).any():
                print(f"Detected NaN in parameter:FBA")
        dataset = torch.stack([torch.stack(sample) for sample in dataset])
        falsedataset = torch.stack([torch.stack(sample) for sample in falsedataset])
        dataset=dataset.transpose(0, 1)
        falsedataset=falsedataset.transpose(0, 1)
        return dataset,falsedataset


    def get_query_diedai(self, ):
        dataset = []
        falsedataset = []
        datapath = './../train/train_res18_new/gedaifea_Calt_Dogs'
        all_files = os.listdir(datapath)
        def sort_key(filename):
            match = re.match(r'Calt_(\d+)_Calt_(\d+)\.pth\.pth_ep(\d+)', filename)
            if match:
                x = int(match.group(1))
                y = int(match.group(2))
                ep = int(match.group(3))
                return (y, x, ep)
            return (float('inf'), float('inf'), float('inf'))

        sorted_files = sorted(all_files, key=sort_key)

        for cname in sorted_files:
            cdata = torch.load(os.path.join(datapath, cname))
            
            b_afea = l2_normalize(cdata['b_afea'].squeeze().squeeze())
            p_fea = l2_normalize(cdata['pfea'].squeeze().squeeze())
            c_fea = l2_normalize(cdata['cfea'].squeeze().squeeze())
            fb_afea = l2_normalize(cdata['fb_afea'].squeeze().squeeze())
            fcfea = l2_normalize(cdata['fcfea'].squeeze().squeeze())
            
            b_afea = pad_to_64_512(b_afea)
            c_fea = pad_to_64_512(c_fea)
            p_fea = pad_to_64_512(p_fea)
            fb_afea = pad_to_64_512(fb_afea)
            fcfea = pad_to_64_512(fcfea)

            if torch.isnan(b_afea).any():
                print(f"Detected NaN in BAFE A: {cname}")
            if torch.isnan(p_fea).any():
                print(f"Detected NaN in PFEA: {cname}")
            if torch.isnan(c_fea).any():
                print(f"Detected NaN in C: {cname}")
            if torch.isnan(fb_afea).any():
                print(f"Detected NaN in FBA: {cname}")

            for i in range(b_afea.shape[0]):
                print(cname,"cname")
                dataset.append((cname, [b_afea[i], c_fea[i], p_fea[i]]))
                falsedataset.append((cname, [fb_afea[i], fcfea[i], p_fea[i]]))

        dataset = torch.stack([torch.stack(sample[1]) for sample in dataset])
        falsedataset = torch.stack([torch.stack(sample[1]) for sample in falsedataset])
        dataset=dataset.transpose(0, 1)
        falsedataset=falsedataset.transpose(0, 1) 
        return dataset,falsedataset

    
    def __getitem__(self, index):
        poscname, pos_sample = self.dataset[index]
        negcname, neg_sample = self.falsedataset[index]
        return poscname,negcname,pos_sample, neg_sample


class SplitDataset:
    def __init__(self, dataset, test_ratio=0.2):
        self.train_dataset, self.test_dataset = self.split_dataset(dataset, test_ratio)

    def split_dataset(self, dataset, test_ratio):

        dataset_size = len(dataset)
        test_size = int(dataset_size * test_ratio)
        train_size = dataset_size - test_size 
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset


def create_dataloader(dataset):
    # dataset = EmbeddingDataset()
    batch_size = 60
    split_dataset = SplitDataset(dataset, test_ratio=0.1)

    train_dataset = split_dataset.train_dataset
    test_dataset = split_dataset.test_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    return  train_loader,test_loader

if __name__ == "__main__":
    dataset = EmbeddingDataset()
    batch_size = 60
    split_dataset = SplitDataset(dataset, test_ratio=0.2)

    train_dataset = split_dataset.train_dataset
    test_dataset = split_dataset.test_dataset

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))