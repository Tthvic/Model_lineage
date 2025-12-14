import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .utils import get_device

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(dataloader), 100. * correct / total

def train_model(model, dataloader, config, device=None):
    if device is None:
        device = get_device()
    
    model = model.to(device)
    
    # Config parameters
    lr = config.get('learning_rate', 0.001)
    num_epochs = config.get('num_epochs', 10)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        # scheduler.step()
        
    return model
