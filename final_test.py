import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,SubsetRandomSampler, ConcatDataset
from torchmetrics import Accuracy

from dataset import PersonalityDataset

from models.mlp import MLPsimple
from models.cnn8 import CNN8simple

dataset_name = 'BFD'
dataset_type = 'gray'


def train_epoch(model,device,dataloader,loss_fn,optimizer,train_accuracy):
    train_loss, train_correct=0.0,0
    model.train()
    for batch in dataloader:
        images, labels = batch['normalized'], batch['label']
        images = images.to(device)
        optimizer.zero_grad()
        output = model(images)
        labels = torch.tensor(labels, dtype=torch.float32, device=output.device)
        loss = loss_fn(output.flatten(), labels.flatten())
        train_accuracy(output, labels.to(torch.int64))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        predictions = torch.where(output>0, 1, 0)
        train_correct += (predictions == labels.to(torch.int64)).sum().item()

    return train_loss, train_correct
  
def valid_epoch(model,device,dataloader,loss_fn,val_accuracy):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    for batch in dataloader:
        images, labels = batch['normalized'], batch['label']
        images = images.to(device)
        output = model(images)
        labels = torch.tensor(labels, dtype=torch.float32, device=output.device)
        loss = loss_fn(output.flatten(),labels.flatten())
        val_accuracy(output, labels.to(torch.int64))
        valid_loss += loss.item() * images.size(0)
        predictions = torch.where(output>0, 1, 0)
        val_correct += (predictions == labels.to(torch.int64)).sum().item()

    return valid_loss, val_correct



LocationConfig_data = f'data/{dataset_name}/{dataset_type}/'

model_name = f'{dataset_name}_{dataset_type}'
params = {}
params['BFD_enc'] = {'batch_norm': True,'batch_size': 16,'dropout': 0.4,'lr': 0.001,'negative_slope': 0.05}
params['BFD_gray'] = {'batch_norm': False,'batch_size': 16,'dropout': 0.4,'lr': 0.001,'negative_slope': 0.1}
params['BFD_rgb'] = {'batch_norm': False,'batch_size': 8,'dropout': 0.0,'lr': 0.00005,'negative_slope': 0.02}
params['ChaLearn_enc'] = {'batch_norm': False,'batch_size': 4,'dropout': 0.3,'lr': 0.001,'negative_slope': 0.1}
params['ChaLearn_gray'] = {'batch_norm': True,'batch_size': 4,'dropout': 0.0,'lr': 0.001,'negative_slope': 0.01}
params['ChaLearn_rgb'] = {'batch_norm': False,'batch_size': 8,'dropout': 0.0,'lr': 0.00005,'negative_slope': 0.1}

epochs = {}
epochs['BFD_enc'] = 10
epochs['BFD_gray'] = 70
epochs['BFD_rgb'] = 24
epochs['ChaLearn_enc'] = 12
epochs['ChaLearn_gray'] = 5
epochs['ChaLearn_rgb'] = 4


train_dataset = PersonalityDataset(Path(LocationConfig_data + 'train/'))
test_dataset = PersonalityDataset(Path(LocationConfig_data + 'test/'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
dataset = ConcatDataset([train_dataset, test_dataset])

m=len(train_dataset)



k=10
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    train_accuracy = Accuracy(threshold=0.0).cuda()
    val_accuracy = Accuracy(threshold=0.0).cuda()

    print('Fold {}'.format(fold + 1))
    criterion = nn.BCEWithLogitsLoss()  
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=params[model_name]['batch_size'], sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=params[model_name]['batch_size'], sampler=test_sampler)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if dataset_type=='enc':
        model = MLPsimple(**params)
    else:
        model = CNN8simple(data_type=dataset_type, dataset=dataset_name, **params[model_name])
    
    
#     model = CNN8simple(**params[model_name])
    # model = MLPsimple(**params)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params[model_name]['lr'])

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[],'train_acc_2':[],'test_acc_2':[]}

    for epoch in range(epochs[model_name]):
        train_loss, train_correct=train_epoch(model,device,train_loader,criterion,optimizer,train_accuracy)
        test_loss, test_correct=valid_epoch(model,device,test_loader,criterion,val_accuracy)

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / (len(train_loader.sampler) * 5) * 100
        train_acc_2 = train_accuracy.compute() * 100
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / (len(test_loader.sampler) * 5) * 100
        test_acc_2 = val_accuracy.compute() * 100

        print("F {} | E:{}/{} Tra Loss:{:.3f} Test Loss:{:.3f} Tra Acc {:.2f}% | {:.2f}% Test Acc {:.2f}% | {:.2f}%".format(
            fold + 1,
            epoch + 1,
            epochs[model_name],
            train_loss,
            test_loss,
            train_acc,
            train_acc_2,
            test_acc,
            test_acc_2
            ))
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_acc_2'].append(train_acc_2.item())
        history['test_acc_2'].append(test_acc_2.item())

    foldperf['fold{}'.format(fold+1)] = history  

torch.save(model,f'model/k_cross/{dataset_name}_{dataset_type}.pt')
a_file = open(f'results/{dataset_name}_{dataset_type}.pkl', 'wb')
pickle.dump(foldperf, a_file)
a_file.close()