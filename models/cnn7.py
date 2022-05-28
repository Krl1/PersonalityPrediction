import torchvision.transforms as transforms
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torch import nn
import torch

class CNN7(pl.LightningModule):
    
    def __init__(
        self,
        lr: float,
        batch_norm: bool,
        negative_slope: float = 0.0,
        dropout: float = 0.4,
        batch_size: int = 128
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.batch_norm = batch_norm
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.batch_size = batch_size
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Conv
        fm_size = 16
        self.conv = nn.Sequential()
        self.conv.add_module(f"conv_0",nn.Conv2d(1, fm_size, (3,3), stride=1, padding='same'))
        if self.batch_norm:
            self.conv.add_module(f"batch_norm_0", nn.BatchNorm2d(fm_size))
        if self.negative_slope == 0.0:
            self.conv.add_module(f"relu_0", nn.ReLU())
        else:
            self.conv.add_module(f"lrelu_0", nn.LeakyReLU(self.negative_slope))
        if self.dropout != 0.0:
            self.conv.add_module(f"dropout_0", nn.Dropout(self.dropout))
        self.conv.add_module(f"max_pool_0", nn.MaxPool2d(2))
        
        for i in range(1, 7):
            self.conv.add_module(f"conv_{i}", nn.Conv2d(fm_size, fm_size*2, (3,3), stride=1, padding='same'))
            if self.batch_norm:
                self.conv.add_module(f"batch_norm_{i}", nn.BatchNorm2d(fm_size*2))
            if self.negative_slope == 0.0:
                self.conv.add_module(f"relu_{i}", nn.ReLU())
            else:
                self.conv.add_module(f"lrelu_{i}", nn.LeakyReLU(self.negative_slope))
            if self.dropout != 0.0:
                self.conv.add_module(f"dropout_{i}", nn.Dropout(self.dropout))
            self.conv.add_module(f"max_pool_{i}", nn.MaxPool2d(2))
            fm_size *= 2
        
        self.conv.add_module(f"conv_7",nn.Conv2d(fm_size, fm_size*2, (3,3), stride=1, padding='same'))
        if self.batch_norm:
            self.conv.add_module(f"batch_norm_7", nn.BatchNorm2d(fm_size*2))
        if self.negative_slope == 0.0:
            self.conv.add_module(f"relu_7", nn.ReLU())
        else:
            self.conv.add_module(f"lrelu_7", nn.LeakyReLU(self.negative_slope))
        if self.dropout != 0.0:
            self.conv.add_module(f"dropout_7", nn.Dropout(self.dropout))
        self.conv.add_module(f"max_pool_7", nn.MaxPool2d(2, ceil_mode=True))
        fm_size *= 2
        
        # Linear
        self.linear = nn.Sequential() 
        self.linear.add_module("flatten", nn.Flatten())
        self.linear.add_module("linear_0", nn.Linear(fm_size, 128))
        if self.batch_norm:
            self.linear.add_module(f"batch_norm_0", nn.BatchNorm1d(128))
        if self.negative_slope ==0.0:
            self.linear.add_module(f"relu_0", nn.ReLU())
        else:
            self.linear.add_module(f"lrelu_0", nn.LeakyReLU(self.negative_slope))
        if self.dropout != 0.0:
            self.conv.add_module(f"dropout_0", nn.Dropout(self.dropout))
        
        self.linear.add_module("linear_1", nn.Linear(128, 32))
        if self.batch_norm:
            self.linear.add_module(f"batch_norm_1", nn.BatchNorm1d(32))
        if self.negative_slope == 0.0:
            self.linear.add_module(f"relu_0", nn.ReLU())
        else:
            self.linear.add_module(f"lrelu_1", nn.LeakyReLU(self.negative_slope))
        if self.dropout != 0.0:
            self.conv.add_module(f"dropout_0", nn.Dropout(self.dropout/2))
        
        self.linear.add_module("linear_2", nn.Linear(32, 5))

    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.linear(x) 
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['normalized'], batch['label']
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)
        loss = self.criterion(y_pred.flatten(), y.flatten())
        train_acc_batch = self.train_accuracy(y_pred, y.to(torch.int64))
        self.log('train_acc_batch', train_acc_batch, on_step=True, on_epoch=False)
        self.log('train_loss_batch', loss.mean(), on_step=True, on_epoch=False)
        
        return {'loss': loss, 'y_pred': y_pred, 'y': y}
    
    def training_epoch_end(self, outputs):
        sum_loss = 0.0
        losses = {'ext':0.0, 'agr':0.0, 'con':0.0, 'neu':0.0, 'ope':0.0}
        for output in outputs:
            sum_loss += output['loss'].item()
            y_pred = output['y_pred']
            y = output['y']
            for i, name in enumerate(['ext','agr','con','neu','ope']):
                losses[name] += nn.BCEWithLogitsLoss()(y_pred[:,i], y[:,i])
        
        for name in (['ext','agr','con','neu','ope']):
            losses[name] /= len(outputs)
            self.log(f'train_loss_{name}_epoch', losses[name])
            
        self.log('train_loss_epoch', sum_loss/len(outputs))
        accuracy = self.train_accuracy.compute()
        self.log('train_acc_epoch', accuracy)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['normalized'], batch['label']
        y_pred = self.forward(x)
        y = torch.tensor(y, dtype=torch.float32, device=y_pred.device)
        loss = [] 
        for i in range(5):
            loss.append(self.criterion(y_pred[:,i], y[:,i]))
        val_acc_batch = self.val_accuracy(y_pred, y.to(torch.int64))
        return loss
    
    def validation_epoch_end(self, outputs):
        outputs = torch.tensor(outputs)
        for i, name in enumerate(['ext','agr','con','neu','ope']):
            self.log(f'val_loss_{name}_epoch', torch.mean(outputs, dim=0)[i])
        self.log('val_loss_epoch', torch.mean(outputs))
        accuracy = self.val_accuracy.compute()
        self.log('val_acc_epoch', accuracy)
