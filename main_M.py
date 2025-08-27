# Define training process
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.optim as optim
from Small_model.test_M_final.network3 import *
####### dataloader824 tests grandparent-grandchild relationship
#### Used to save a final model to verify grandparent-grandchild relationship
from Small_model.test_M_final.dataloader824 import create_dataloader
# from dataloader6 import pathdict
# from loss import *
from torch.utils.data import TensorDataset, DataLoader
from Small_model.test_M_final.load_mobilenet import *
import torch
import torch.nn as nn
import torch.nn.functional as F
T=0.3

class TripletLoss(nn.Module):
    def __init__(self, margin=0.1):
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

# Save results to .txt file
def save_results_txt(results_dict, epoch, name):
    txt_path = os.path.join('./ftsims/', f'Dogs{epoch}{name}.txt')
    with open(txt_path, 'w') as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")



def testparent_childandgrandparent_child(encnet, prenet, save_dir, epoch):
    # Initialize result dictionary
    results = {"parent_child": [], "grandparent_child": []}  # Store complete data of parent_child and grandparent-grandchild
    encnet.eval()
    prenet.eval()
    model_save_path = os.path.join(save_dir, f"framework_models_epoch{epoch}.pth")
    save_path = os.path.join(save_dir, f"framework_epoch{epoch}.pth")

    # Load model parameters from local
    model_load_path = os.path.join(save_dir, f"framework_models_epoch{epoch}.pth")
    device = next(encnet.parameters()).device
    checkpoint = torch.load(model_load_path, map_location=device)
    encnet.load_state_dict(checkpoint["encnet_state_dict"])
    prenet.load_state_dict(checkpoint["prenet_state_dict"])
    print(f"[Epoch {epoch}] Model parameters loaded from {model_load_path}")

    # Load saved results from local
    reload_results = torch.load(save_path)
    print(f"[Epoch {epoch}] Reloaded saved data")
    
    # Reload data into the model and test
    test_results = {"parent_child": [], "grandparent_child": [],"non_lineage": []}
    for category, data_list in reload_results.items():
        for data in data_list:
            # Reload features
            cfea= data["cfea"].to('cuda:0') # Ensure on the correct device
            pfea =data["pfea"].to('cuda:0')
            b_afea = data["b_afea"].to('cuda:0')
            # Encode features
            enccfea = encnet(cfea.unsqueeze(0))  # Add batch dimension
            enb_apfea = encnet(b_afea.unsqueeze(0))
            # Recompute similarity
            pknow = encnet(pfea.unsqueeze(0))
            fsumknow = prenet(enb_apfea, enccfea)
            sim = torch.nn.functional.cosine_similarity(pknow, fsumknow).item()
            # Save test results
            test_results[category].append({
                "posname": data["posname"],
                "similarity": sim
            })

    # Save test results to local file
    test_save_path = os.path.join(save_dir, f"framework_test_epoch{epoch}.pth")
    torch.save(test_results, test_save_path)
    # print("test_resu_parent_child",test_results["parent_child"])
    # print("test_resu_grandparent_child",test_results["grandparent_child"])
    for category in ["parent_child", "grandparent_child", "non_lineage"]:
        sims = test_results[category]
        if len(sims) == 0:
            print(f"No results for {category}")
            continue
        limit = min(10, len(sims))
        for i in range(limit):
            print(sims[i]['similarity'], f"test_results_{category}")
    # print(f"[Epoch {epoch}] Test results saved to: {test_save_path}")

    return reload_results, test_results



def test_fpr(encnet, prenet, test_loader, contrastive_loss_fn, epoch, name):
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

    # Initialize containers per category: positives/negatives similarities, correct counts and totals
    results = {
        "Calt": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
        "Flowers": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
        "TINY": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
    }

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradients during testing
        for i, (posname, negname, data, falsedata) in enumerate(test_loader):
            b_afea = data[0].to('cuda:0')
            cfea = data[1].to('cuda:0')
            pfea = data[2].to('cuda:0')
            fb_afea = falsedata[0].to('cuda:0')
            fcfea = falsedata[1].to('cuda:0')
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
            print(pos_similarity,neg_similarity,"pos_neg_similarity")
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

                if pos_similarity[j] >= T:
                    results[category]["correct_positive"] += 1
                if neg_similarity[j] >=T:
                    results[category]["correct_negative"] += 1
                results[category]["total"] += 1

                # Store posname and embedding information
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
        print(results['Flowers'].keys() ,"lenfls")
        # Print results
    for category in results:
        if category == "overall":
            continue

        total = results[category]["total"]
        correct_positive = results[category]["correct_positive"]
        correct_negative = results[category]["correct_negative"]

        accuracy = (correct_positive) / total if total > 0 else 0.0
        fpr = correct_negative / total if total > 0 else 0.0

        results[category]["accuracy"] = accuracy
        results[category]["fpr"] = fpr

    # Print results
    for category in results:
        if category == "overall":
            continue
        print(f"Category: {category}")
        print(f"Accuracy: {results[category]['accuracy']:.4f}")
        print(f"False Positive Rate: {results[category]['fpr']:.4f}")

    print("save")
    torch.save(results,os.path.join('./grandparent_childsims',f'{epoch}'+'.pth')) 

    return avg_test_loss, overall_accuracy


def main():

    index = 0  
    batch_size = 40
    encnet = TransformerEncoder(feat_dim=1280)
    encnet = encnet.to('cuda:0')
    prenet = VectorRelationNet(256)
    prenet = prenet.to('cuda:0')
    # train_loader,test_loader,test_loader2=create_dataloader()
    # optimizer = optim.Adam(list(encnet.parameters()) + list(prenet.parameters()), lr=0.001)
    # triplet_loss_fn = TripletLoss(margin=0.2)
    # num_epochs = 2
    # def train():
    #     for epoch in range(num_epochs):
    #         encnet.train()  # Set to training mode
    #         prenet.train()
    #         for i, (_,_,data, falsedata) in enumerate(train_loader):
    #             b_afea,cfea,pfea=data[0].to('cuda:0'),data[1].to('cuda:0'),data[2].to('cuda:0')
    #             encb_afea,enccfea,encpfea=encnet(b_afea),encnet(cfea),encnet(pfea)
    #             pknow=encpfea
    #             sumknow=prenet(encb_afea,enccfea)
    #             fb_afea,fcfea,fpfea=falsedata[0].to('cuda:0'),falsedata[1].to('cuda:0'),falsedata[2].to('cuda:0')
    #             fencb_afea, fenccfea=encnet(fb_afea), encnet(fcfea)
    #             fsumknow=prenet(fencb_afea,fenccfea)
    #           
    #             loss1 = triplet_loss_fn(pknow, sumknow, fsumknow)  
    #             # loss2 = triplet_loss_fn(fpfea,pknow, fsumknow)  
    #             loss3 = triplet_loss_fn(sumknow, pknow, fsumknow)  
    #          
    #             loss = loss1 + loss3
    #             # loss = triplet_loss_fn(pknow, sumknow, fsumknow)
    #             # Backpropagation and optimization
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             # Compute cosine similarity
    def test():
        epoch=1
        testparent_childandgrandparent_child(encnet, prenet,save_dir="Small_model/test_M_final/framework",epoch=epoch)
    test()

if __name__ == "__main__":
    main()