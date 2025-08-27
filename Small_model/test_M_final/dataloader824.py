import os
import torch
import re
from torch.utils.data import Dataset,TensorDataset, DataLoader, random_split, Subset
import random


def simplify_filename(file_name):
    """
    Simplify file name by removing redundant '.pth' before extension.

    :param file_name: Input file name to process.
    :return: Simplified file name.
    """
    # Split the file name into parts
    parts = file_name.split(".")

    if len(parts) >= 3 and parts[-2] == "pth":  # Check if there are redundant 'pth' parts
        # Reconstruct the simplified file name
        simplified_name = f"{'.'.join(parts[:-2])}.{parts[-1]}"
        return simplified_name
    else:
        return file_name  # Return the original name if no simplification is needed


# L2 normalization function
# def l2_normalize(tensor):
#     norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
#     return tensor / (norm + 1e-8)

def l2_normalize(tensor, min_val=None, max_val=None):
    """
    Placeholder normalization. Currently returns input tensor unchanged.
    Add normalization if needed; keep NaN check for safety.
    """
    if torch.isnan(tensor).any():
        print("Detected NaN in tensor")
    return tensor


def get_p_names(file_name):
    # Use regex to match the parent model name segment
    match = re.search(r"Calt_\d+_(.+?)$", file_name)
    if match:
        return match.group(1)  # Return parent model name
    else:
        raise ValueError(f"Invalid file name format: {file_name}")


 # Check and pad features to expected shapes
def pad_to_64_512(tensor):
    if tensor.shape[0] == 64:
        return tensor.reshape(8, 8, 512)
    elif tensor.shape[0] == 36:
        padded_tensor = torch.zeros((6, 8, 1280), dtype=tensor.dtype, device=tensor.device)
        padded_tensor[:, :6, :] = tensor.reshape(6, 6, 1280)
        return padded_tensor
    else:
        raise ValueError("Unsupported tensor shape for padding")



class EmbeddingDataset(Dataset):
    def __init__(self, trainflag=True,test_WP=False,test_grandparent_child=False):
        self.trainflag=trainflag
        self.grandparent_childflag=test_grandparent_child
        self.WPflag=test_WP
        self.datapath='./../train/train_mobilenet/Feas'
        if self.grandparent_childflag:
            self.dataset,self.falsedataset=self.load_data_grandparent_child()
        elif self.WPflag:
            self.dataset,self.falsedataset=self.load_data_WP()
        else:
            self.flowers_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
                    for name in os.listdir('./../train/train_mobilenet/Feas/Flowers')]
            self.calt_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
                for name in os.listdir('./../train/train_mobilenet/Feas/Calt')]
            self.tiny_models=[name.split('_epoch')[0] + '.pth' if '_epoch' in name else name 
                for name in os.listdir('./../train/train_mobilenet/Feas/TINYIMG')]
            self.flowers_models=list(set(self.flowers_models))
            self.calt_models=list(set(self.calt_models))
            self.tiny_models=list(set(self.tiny_models))
            self.train_set=[]
            self.test_set=[]
            for model_group in [self.flowers_models, self.calt_models, self.tiny_models]:
                random.shuffle(model_group)
                split_idx = int(0.6 * len(model_group)) if len(model_group) > 1 else 0
                self.train_set.extend(model_group[:split_idx])
                self.test_set.extend(model_group[split_idx:])
            # Debug prints removed for cleanliness
            self.dataset,self.falsedataset = self.load_data_total()

    def __len__(self):
        """Return total number of samples."""
        return len(self.dataset)

    def load_data_total(self,):
        if self.grandparent_childflag==True:
            dataset,falsedataset=self.load_data_grandparent_child()
        elif self.WPflag==True:
            dataset,falsedataset=self.load_data_WP()
        else:
            dataset=[]
            falsedataset=[]
            nonCalt_fea_path='Calt'
            dataset,falsedataset=self.load_data(nonCalt_fea_path,dataset,falsedataset)
            nonDogs_fea_path='TINYIMG'
            dataset,falsedataset=self.load_data(nonDogs_fea_path,dataset,falsedataset)
            nonFlowers_fea_path='Flowers'
            dataset,falsedataset=self.load_data(nonFlowers_fea_path,dataset,falsedataset)

        return dataset,falsedataset


    def load_data(self,eachdatapath,dataset,falsedataset):
        datapath = os.path.join(self.datapath,eachdatapath)
        for cname in os.listdir(datapath):  ##### order: b_a, child, parent
            ccname=cname.split('_epoch')[0] + '.pth' 
            if self.trainflag==True and ccname in self.test_set:
                continue
            if self.trainflag==False and ccname in self.train_set:
                print(ccname,"ccname")
                continue          
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
            # Optional NaN checks (silenced by default for speed)
            # if torch.isnan(b_afea).any():
            #     print("Detected NaN: b_afea")
            # if torch.isnan(p_fea).any():
            #     print("Detected NaN: p_fea")
            # if torch.isnan(c_fea).any():
            #     print("Detected NaN: c_fea")
            # if torch.isnan(fb_afea).any():
            #     print("Detected NaN: fb_afea")

        return dataset,falsedataset

    def load_data_grandparent_child(self,):
        dataset=[]
        falsedataset=[]
        Calt_fea_path='Calt'
        dataset,falsedataset=self.load_data_other(Calt_fea_path,dataset,falsedataset,flag="grandparent_child")
        Dogs_fea_path='TINYIMG'
        dataset,falsedataset=self.load_data_other(Dogs_fea_path,dataset,falsedataset,flag="grandparent_child")
        Flowers_fea_path='Flowers'
        dataset,falsedataset=self.load_data_other(Flowers_fea_path,dataset,falsedataset,flag="grandparent_child")
        # print(len(dataset), len(falsedataset), "lenfal")
        return dataset,falsedataset
 

    def load_data_WP(self,):
        dataset=[]
        falsedataset=[]
        Calt_fea_path='Calt'
        dataset,falsedataset=self.load_data_other(Calt_fea_path,dataset,falsedataset,flag="WP")
        Dogs_fea_path='TINYIMG'
        dataset,falsedataset=self.load_data_other(Dogs_fea_path,dataset,falsedataset,flag="WP")
        Flowers_fea_path='Flowers'
        dataset,falsedataset=self.load_data_other(Flowers_fea_path,dataset,falsedataset,flag="WP")
        print(len(dataset),len(falsedataset),"lenfal")
        return dataset,falsedataset


    def load_data_other(self,eachdatapath,dataset,falsedataset,flag):
        if flag=='grandparent_child':
            path='./../train/train_mobilenet/grandparent_childFea'
        else:
            path='./../train/train_mobilenet/WPFEA'
        datapath = os.path.join(path,eachdatapath)
        ccnames=[]
        for cname in os.listdir(datapath):
            ccname=cname.split('_epoch')[0]     
            # print(ccname,'ccname')
            if ccname in ccnames:
                continue
            ccnames.append(ccname) 
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

        return dataset,falsedataset


    def __getitem__(self, index):
        poscname, pos_sample = self.dataset[index]
        negcname, neg_sample = self.falsedataset[index]
        return poscname,negcname,pos_sample, neg_sample   

class SplitDataset:
    def __init__(self, dataset, test_ratio=0.8):
        self.train_dataset, self.test_dataset = self.split_dataset(dataset, test_ratio)

    def split_dataset(self, dataset, test_ratio):
        # Compute lengths for train and test sets
        dataset_size = len(dataset)
        test_size = int(dataset_size * test_ratio)
        train_size = dataset_size - test_size  # Use random_split to split dataset
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        return train_dataset, test_dataset


def create_dataloader():
    batch_size=128
    train_dataset = EmbeddingDataset(trainflag=True)  # Create dataset instance
    test_dataset=EmbeddingDataset(trainflag=False,test_grandparent_child=True)
    test_dataset2=EmbeddingDataset(trainflag=False,test_grandparent_child=False)
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loader2 = DataLoader(test_dataset2, batch_size=batch_size, shuffle=False)
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    return  train_loader,test_loader,test_loader2

if __name__ == "__main__":
    dataset = EmbeddingDataset()
    batch_size = 60
    # Split dataset into train and test
    split_dataset = SplitDataset(dataset, test_ratio=0.2)

    # Get train and test sets
    train_dataset = split_dataset.train_dataset
    test_dataset = split_dataset.test_dataset

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))