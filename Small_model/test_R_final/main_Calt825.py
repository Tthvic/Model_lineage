##### Test intergenerational lineage similarity
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.optim as optim
from network3 import *
from dataloader810 import create_dataloader,EmbeddingDataset,pathdict
from loss import *
from torch.utils.data import TensorDataset, DataLoader

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


def test(encnet, prenet, test_loader, contrastive_loss_fn, epoch, name):
    print("testparent_child")
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

    # Initialize storage for each category: pos/neg similarities, correct counts and totals
    results = {
        "Calt": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
        "Flowers": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
        "Dogs": {"all_pos_similarities": [], "all_neg_similarities": [], "correct_positive": 0, "correct_negative": 0, "total": 0},
    }

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient computation during testing
        for i, (posname, negname, data, falsedata) in enumerate(test_loader):
            if i>0:
                break
            # data: tuple containing (b_afea, cfea, pfea)
            # falsedata: tuple containing (fb_afea, fcfea, pfea)
            b_afea = data[0].to('cuda:0')
            cfea = data[1].to('cuda:0')
            pfea = data[2].to('cuda:0')
            fb_afea = falsedata[0].to('cuda:0')
            fcfea = falsedata[1].to('cuda:0')
            fpfea = falsedata[2].to('cuda:0')

            # Forward pass through encnet
            encb_afea = encnet(b_afea)
            enccfea = encnet(cfea)
            encpfea = encnet(pfea)
            fencb_afea = encnet(fb_afea)
            fenccfea = encnet(fcfea)

            # Forward pass through prenet
            pknow = encpfea
            sumknow = prenet(encb_afea, enccfea)  # Positive pair
            fsumknow = prenet(fencb_afea, fenccfea)  # Negative pair

            # Compute cosine similarity
            pos_similarity = torch.nn.functional.cosine_similarity(pknow, sumknow).cpu()
            neg_similarity = torch.nn.functional.cosine_similarity(pknow, fsumknow).cpu()
            print(pos_similarity,neg_similarity,"poss")
            batch_size = b_afea.shape[0]
            num_batches += 1

            # Iterate samples in batch and categorize by filename
            for j in range(batch_size):
                # Assume pos/neg share the same filename, use posname[j]
                fname = posname[j]
                if "Calt" in fname:
                    category = "Calt"
                elif "Flowers" in fname:
                    category = "Flowers"
                elif "Dogs" in fname:
                    category = "Dogs"
                else:
                    category = "Other"
                    print("other_category")
                # Save similarity
                results[category]["all_pos_similarities"].append(pos_similarity[j].item())
                results[category]["all_neg_similarities"].append(neg_similarity[j].item())
                # Update counts by threshold
                if pos_similarity[j] >0.7:
                    results[category]["correct_positive"] += 1
                if neg_similarity[j] >0.7:
                    results[category]["correct_negative"] += 1
                results[category]["total"] += 1

    # Compute overall accuracy (positive correct / total)
    overall_total = sum(results[cat]["total"] for cat in results if results[cat]["total"] > 0)
    overall_correct = sum(results[cat]["correct_positive"] for cat in results if results[cat]["total"] > 0)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    results["overall"] = {"accuracy": overall_accuracy, "total": overall_total, "correct": overall_correct}
    print(overall_accuracy,"over_accy")
    # If needed, accumulate contrastive loss here (not computed)
    avg_test_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    return avg_test_loss, overall_accuracy




def framework( encnet, prenet, save_dir, epoch):
    # Initialize result containers
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
    print(reload_results.keys(),"kkk")
    print(f"[Epoch {epoch}] Reloaded saved data")
    test_results = {"parent_child": [], "grandparent_child": [] ,'non_lineage':[]}
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
                "similarity": sim
            })

    # test_save_path = os.path.join(save_dir, f"framework_test_epoch{epoch}.pth")
    # torch.save(test_results, test_save_path)
    for i in range(10):
        print(test_results["parent_child"][i]['similarity'],"test_results_parent_child")
        print(test_results["grandparent_child"][i]['similarity'],"test_results_grandparent_child")
        print(test_results["non_lineage"][i]['similarity'],"test_results_grandparent_child")

    return reload_results, test_results



def main():

    # dataset = EmbeddingDataset()  # Create dataset instance
    index = 0  # Assume extracting batch size 16 with start index
    batch_size = 100
    encnet = TransformerEncoder(feat_dim=512)
    encnet = encnet.to('cuda:0')
    prenet = VectorRelationNet(256)
    prenet = prenet.to('cuda:0')
    ftdatas=[]
    ftfalsedatas=[]
    def train():
        train_loader,test_loader=create_dataloader(dataset)
        # Initialize optimizer for the networks
        optimizer = optim.Adam(list(encnet.parameters()) + list(prenet.parameters()), lr=0.0001)
        triplet_loss_fn = TripletLoss(margin=0.4)
        num_epochs = 1000
        grandparent_childdata,falsegrandparent_childdata=dataset.get_query_grandparent_child(pathdict['Calt_sun'])
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
                loss1 = triplet_loss_fn(pknow, sumknow, fsumknow)
                loss3 = triplet_loss_fn(sumknow, pknow, fsumknow)
                loss = loss1 + loss3
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    def test():
        epoch=1
        save_dir='./framework'
        framework( encnet, prenet, save_dir, epoch)
    test()
if __name__ == "__main__":
    main()