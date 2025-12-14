# 定义训练过程
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.optim as optim
from network3 import *
from dataloader1213 import create_dataloader
# from dataloader6 import create_dataloader
# from dataloader6 import pathdict
# from dataloader9 import create_dataloader
from loss import *
from torch.utils.data import TensorDataset, DataLoader
from load_mobilenet import *


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize embeddings to improve numerical stability
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Compute pairwise distances
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)

        # Compute triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)

        return torch.mean(loss)

# 保存结果到 .txt 文件
def save_results_txt(results_dict, epoch, name):
    txt_path = os.path.join('./ftsims/', f'Dogs{epoch}{name}.txt')
    with open(txt_path, 'w') as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")




def test(encnet, prenet, test_loader, contrastive_loss_fn, epoch, name):
    """
    Test function to evaluate the model on the test dataset and compare positive vs negative pair outputs.
    The results are stored separately according to the file name (cname) containing 'Calt', 'Flowers' or 'Dogs'.

    :param encnet: The TransformerEncoder network.
    :param prenet: The VectorRelationNet network.
    :param test_loader: DataLoader for the test dataset. Each batch returns
                        (posname, negname, data, falsedata), where posname/negname are file name lists.
    :param contrastive_loss_fn: Contrastive loss function.
    :param epoch: Current epoch (for saving file naming).
    :param name: Name tag (for saving file naming).
    :return: Dummy test loss (0.0) and overall accuracy.
    """
    encnet.eval()  # Set encnet to evaluation mode
    prenet.eval()  # Set prenet to evaluation mode

    # 初始化存储结果的字典，每个类别保存正/负相似性、正确计数和总数
    results = {
        "Calt": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
        "Flowers": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
        "TINY": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
    }

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation during testing
        for i, (posname, negname, data, falsedata) in enumerate(test_loader):
            b_afea = data[0].to('cuda:0')
            cfea = data[1].to('cuda:0')
            pfea = data[2].to('cuda:0')
            fb_afea = falsedata[0].to('cuda:0')
            fcfea = falsedata[1].to('cuda:0')
            # fpfea = falsedata[2].to('cuda:0')
            # print(b_afea.shape,cfea.shape,"shape")
            encb_afea = encnet(b_afea)
            enccfea = encnet(cfea)
            encpfea = encnet(pfea)
            fencb_afea = encnet(fb_afea)
            fenccfea = encnet(fcfea)

            pknow = encpfea
            sumknow = prenet(encb_afea, enccfea)  # Positive pair
            fsumknow = prenet(fencb_afea, fenccfea)  # Negative pair

            pos_similarity = torch.nn.functional.cosine_similarity(pknow, sumknow).cpu()
            neg_similarity = torch.nn.functional.cosine_similarity(pknow, fsumknow).cpu()
            batch_size = b_afea.shape[0]
            num_batches += 1

            for j in range(batch_size):
                fname = posname[j]
                if "Calt" in fname:
                    category = "Calt"
                elif "Flowers" in fname:
                    category = "Flowers"
                elif "TINY" in fname:
                    category = "TINY"
                else:
                    category = "Other"
                    print("other_category")

                results[category]["all_pos_similarities"].append(pos_similarity[j].item())
                results[category]["all_neg_similarities"].append(neg_similarity[j].item())
                # print(neg_similarity,"negsim")
                if pos_similarity[j] > 0.3:
                    results[category]["correct_positive"] += 1
                else:
                    results[category]["correct_negative"] += 1
                results[category]["total"] += 1

                # 存储 posname 和 embedding 信息
                if "pos_embeddings" not in results[category]:
                    results[category]["pos_embeddings"] = []
                    results[category]["pos_names"] = []
                    results[category]["sum_embeddings"] = []
                    results[category]["fsum_embeddings"] = []
                    results[category]["cfea"] = []
                    results[category]["b_afea"] = []
                    results[category]["fb_afea"] = []
                    results[category]["pfea"] = []
            
                results[category]["pos_embeddings"].append(pknow[j].cpu().numpy())  
                results[category]["sum_embeddings"].append(sumknow[j].cpu().numpy())
                results[category]["fsum_embeddings"].append(fsumknow[j].cpu().numpy())
                results[category]["pos_names"].append(fname)
                results[category]["cfea"].append(torch.mean(cfea[j],dim=0).cpu().numpy())  
                results[category]["b_afea"].append(torch.mean(b_afea[j],dim=0).cpu().numpy())
                results[category]["pfea"].append(torch.mean(pfea[j],dim=0).cpu().numpy())
                results[category]["fb_afea"].append(torch.mean(fb_afea[j],dim=0).cpu().numpy())
                # print(torch.mean(cfea[j],dim=0).shape,"shape")
        overall_total = sum(results[cat]["total"] for cat in results if results[cat]["total"] > 0)
        overall_correct = sum(results[cat]["correct_positive"] for cat in results if results[cat]["total"] > 0)
        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        results["overall"] = {"accuracy": overall_accuracy, "total": overall_total, "correct": overall_correct}
        avg_test_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(overall_accuracy,"lenfls")
        for key, value in results.items():
            save_path = os.path.join('./sims/', str(epoch) + f'{key}_fuzi.pth')
            torch.save(results[key], save_path)

        return avg_test_loss, overall_accuracy




def main():

    index = 0  # 假设你想提取批次大小为 16 的数据，并且当前批次的起始位置为 index
    batch_size = 40
    encnet = TransformerEncoder(feat_dim=1280)
    encnet = encnet.to('cuda:0')
    prenet = VectorRelationNet(256)
    prenet = prenet.to('cuda:0')
    train_loader,test_loader=create_dataloader()
    # Initialize optimizer for the networks
    optimizer = optim.Adam(list(encnet.parameters()) + list(prenet.parameters()), lr=0.00005)
    triplet_loss_fn = TripletLoss(margin=0.2)
    num_epochs = 1000
    for epoch in range(num_epochs):
        encnet.train()  # Set to training mode
        prenet.train()
        for i, (_,_,data, falsedata) in enumerate(train_loader):
            b_afea,cfea,pfea=data[0].to('cuda:0'),data[1].to('cuda:0'),data[2].to('cuda:0')
            encb_afea,enccfea,encpfea=encnet(b_afea),encnet(cfea),encnet(pfea)
            pknow=encpfea
            sumknow=prenet(encb_afea,enccfea)
            fb_afea,fcfea,fpfea=falsedata[0].to('cuda:0'),falsedata[1].to('cuda:0'),falsedata[2].to('cuda:0')
            fencb_afea, fenccfea=encnet(fb_afea), encnet(fcfea)
            fsumknow=prenet(fencb_afea,fenccfea)
            # 交叉计算三元组损失
            loss1 = triplet_loss_fn(pknow, sumknow, fsumknow)  # 标准三元组
            # loss2 = triplet_loss_fn(fpfea,pknow, fsumknow)  # 交换负样本
            loss3 = triplet_loss_fn(sumknow, pknow, fsumknow)  # 交换 anchor
            # 计算最终损失（可以使用加权平均）
            loss = loss1 + loss3
            # loss = triplet_loss_fn(pknow, sumknow, fsumknow)
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Compute cosine similarity
            # pos_similarity = torch.nn.functional.cosine_similarity(pknow, sumknow).cpu().detach().numpy()
            # neg_similarity = torch.nn.functional.cosine_similarity(pknow, fsumknow).cpu().detach().numpy()
            # Print similarities
            # print(f"Positive Similarity: {pos_similarity}, Negative Similarity: {neg_similarity}, nnegsim")
        if epoch%1==0:
                test(encnet, prenet, test_loader, 1, epoch,2)
                # tnames,fnames,tdata, fdata=ftdatas[ll],ftfalsedatas[ll]
                # test_ft(encnet, prenet, tdata, fdata, epoch, 'ftepoch'+str(ll*2),tnames,fnames)
        # Optionally, print or log the loss to track training progress
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()