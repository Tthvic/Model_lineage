import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import sys
import os
import glob
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../train/train_mobilenet')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../train/train_res18')))
from train_caltech import get_cmodel3, set_seed,get_data_loader,oriclass
from load_mobilenet import get_p_names,load_pmodels,load_initmodels,load_cmodels,dimdict



dimdict = {'Calt_P': 6, 'Calt_C': 4, 'TINY_P': 6, 'TINY_C': 2, 'Flowers_P': 6, 'Flowers_C': 3,
           'Dog_S': 2, 'Air_S': 3, 'DTD_S': 2,'M_S':2, 'TINYIMG_P': 6, 'TINYIMG_C': 2  }


class DistillNet(nn.Module):
    def __init__(self, num_classes):
        super(DistillNet, self).__init__()
        self.features = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 128 * 16 * 16 = 32768
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    return alpha * soft_loss + (1. - alpha) * ce_loss


def synthesize_images(teacher_model, num_images=2000, image_shape=(3, 128, 128), device='cuda:0', iterations=100, batch_size=64):
    teacher_model.eval()
    synth_images = []
    synth_labels = []
    for _ in range(num_images // batch_size):
        images = torch.randn(batch_size, *image_shape, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([images], lr=0.1)
        for it in range(iterations):
            optimizer.zero_grad()
            outputs = teacher_model(images)
            # DeepInversion目标：最大化熵
            loss = -torch.mean(torch.sum(torch.nn.functional.softmax(outputs, dim=1) * torch.log_softmax(outputs, dim=1), dim=1))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = teacher_model(images)
            labels = torch.argmax(logits, dim=1)
            synth_images.append(images.detach().cpu())
            synth_labels.append(labels.detach().cpu())
    synth_images = torch.cat(synth_images, dim=0)
    synth_labels = torch.cat(synth_labels, dim=0)
    return synth_images, synth_labels


def train_distill(student_model, teacher_model, train_loader, test_loader, device, epochs=10):
    optimizer = optim.Adam(student_model.parameters(), lr=1e-4)
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    for epoch in range(epochs):
        student_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            student_logits = student_model(images)
            loss = distillation_loss(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            student_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = student_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
    torch.save(student_model.state_dict(), './distilled_model.pth')
    return student_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    distilled_model_save_dir = './distilled_model3_synth'
    model_files = sorted(glob.glob('./../../train/train_mobilenet/Pmodels/Calt_*.pth'))
    for model_path in model_files:
        if 'init' in model_path:
            continue
        basename = os.path.basename(model_path)
        idx = int(basename.split('_')[1].split('.')[0])
        # 交换结构：teacher为DistillNet，从distilled_model3_synth加载
        teacher_model_path = os.path.join(distilled_model_save_dir, f'Calt_{idx}.pth')
        teacher_model = DistillNet(num_classes=6)
        if os.path.exists(teacher_model_path):
            teacher_model.load_state_dict(torch.load(teacher_model_path))
            print(f"Loaded teacher model from {teacher_model_path}")
        else:
            print(f"Teacher model not found: {teacher_model_path}, using random init.")
        teacher_model.to(args.device)
        teacher_model.eval()

        # student为mobilenet结构，初始参数initFlag=True
        student_model = load_pmodels(basename, 'Calt_P', 'P', initFlag=True)
        student_model.to(args.device)

        # 合成数据
        # synth_images, synth_labels = synthesize_images(teacher_model, num_images=2000, image_shape=(3, 128, 128), device=args.device)
        # synth_dataset = torch.utils.data.TensorDataset(synth_images, synth_labels)
        # train_loader = torch.utils.data.DataLoader(synth_dataset, batch_size=64, shuffle=True)
        # test_loader = torch.utils.data.DataLoader(synth_dataset, batch_size=64, shuffle=False)
        select_classes=oriclass
        # idx=int(basename.split('_')[1].split('.')[0])
        select_class=select_classes[idx]
        train_loader,test_loader=get_data_loader(select_class)
        save_dir = './distilled_model3_synth_fans'

        # torch.save(student_model.state_dict(), os.path.join(save_dir, f'Calt_{idx}_init.pth'))
 
        student_model = train_distill(student_model, teacher_model, train_loader, test_loader, args.device)
        torch.save(student_model.state_dict(), os.path.join(save_dir, f'Calt_{idx}.pth'))