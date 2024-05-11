import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch.nn.functional as F


# Helper Functions
def split_labeled_embeddings(embeddings, labels, test_size=0.1, val_size=0.1, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(embeddings, labels, test_size=test_size, random_state=random_state)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def cluster_counts(labels):
    clusters = {}
    for label in labels:
        clusters[label] = clusters.get(label, 0) + 1
    return clusters

def cluster_weights(labels):
    counts = cluster_counts(labels)
    total_samples = len(labels)
    weights_dict = {label: total_samples / count for label, count in counts.items()}
    max_weight = max(weights_dict.values())
    weights_dict = {label: weight / max_weight for label, weight in weights_dict.items()}
    return weights_dict

def calculate_accuracy(y_true, y_pred):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float().sum()
    return correct / y_true.shape[0]

# Neural Network Definition
class DeeperClassifierNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperClassifierNN, self).__init__()
        self.fc1 = nn.Linear(768, 1536)
        self.bn1 = nn.BatchNorm1d(1536)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1536, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = F.elu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x

import datetime

def train_network(X_train, y_train, X_val, y_val, num_classes, config):
    # Read config parameters
    lr = config['nn']['learning_rate']
    epochs = config['nn']['epochs']
    batch_size = config['nn']['batch_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Compute weights based on training labels
    weight_dict = cluster_weights(y_train)
    sorted_weights = [weight_dict.get(i-1, 1.0) for i in range(num_classes)]

    
    # Adjust labels to make -1 a 0 class
    y_train = torch.tensor([label + 1 for label in y_train], dtype=torch.long).to(device)
    y_val = torch.tensor([label + 1 for label in y_val], dtype=torch.long).to(device)

    #Adjusted model creation to handle multiple/single gpus
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(DeeperClassifierNN(num_classes))
    else:
        print("Using single GPU or CPU")
        model = DeeperClassifierNN(num_classes)

    
    model.to(device)
    weights_tensor = torch.FloatTensor(sorted_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float).to(device), y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float).to(device), y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get current time to append to checkpoint files
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += calculate_accuracy(labels, outputs)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += calculate_accuracy(labels, outputs)
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save a checkpoint at the end of specified epochs or the last epoch
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            dataset_name = config['data']['dataset']
            checkpoint_path = f'{dataset_name}_model_epoch_{epoch+1}_{current_time}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return model




def evaluate_network(model, X_test, y_test, output_file="evaluation_results.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    y_test = torch.tensor([label + 1 for label in y_test], dtype=torch.long).to(device)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    test_loss, test_acc = 0, 0
    all_preds, all_labels = [], []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()
            test_acc += calculate_accuracy(labels, outputs).item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())  # Move predictions back to CPU for numpy conversion
            all_labels.extend(labels.cpu().numpy())  # Move labels back to CPU for numpy conversion

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    
    # Print to console
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    

    report = classification_report(all_labels, all_preds, output_dict=True)
    report_str = classification_report(all_labels, all_preds)
    print(report_str)

    with open(output_file, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")
        f.write(report_str)
        f.write("\nJSON format:\n")
        json.dump(report, f, indent=4)


    return test_loss, test_acc, report
