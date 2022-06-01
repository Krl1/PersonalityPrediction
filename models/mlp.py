import torchvision.transforms as transforms
from torch.nn import functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl
from torch import nn
import torch

class MLP(pl.LightningModule):
    
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
        self.train_accuracy = Accuracy(threshold=0.0)
        self.val_accuracy = Accuracy(threshold=0.0)
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Linear
        self.linear = nn.Sequential() 
        fm_size = 128
        i=0
        while fm_size != 8:
            self.linear.add_module(f"linear_{i}", nn.Linear(fm_size, fm_size//2))
            if self.batch_norm:
                self.linear.add_module(f"batch_norm_{i}", nn.BatchNorm1d(fm_size//2))
            if self.negative_slope ==0.0:
                self.linear.add_module(f"relu_{i}", nn.ReLU())
            else:
                self.linear.add_module(f"lrelu_{i}", nn.LeakyReLU(self.negative_slope))
            if self.dropout != 0.0:
                self.linear.add_module(f"dropout_{i}", nn.Dropout(self.dropout))
            fm_size = fm_size//2
            i+=1
 
        self.linear.add_module(f"linear_{i}", nn.Linear(fm_size, 5))

    
    def forward(self, x):
#         x = x.permute(0, 3, 1, 2)
        x = self.linear(x) 
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['original'], batch['label']
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
        x, y = batch['original'], batch['label']
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

        
class MLPsimple(nn.Module):
    
    def __init__(
        self,
        lr: float,
        batch_norm: bool,
        negative_slope: float = 0.0,
        dropout: float = 0.4,
        batch_size: int = 128
    ):
        super(MLPsimple, self).__init__()
        
        self.lr = lr
        self.batch_norm = batch_norm
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.batch_size = batch_size
        
        # Linear
        self.linear = nn.Sequential() 
        fm_size = 128
        i=0
        while fm_size != 8:
            self.linear.add_module(f"linear_{i}", nn.Linear(fm_size, fm_size//2))
            if self.batch_norm:
                self.linear.add_module(f"batch_norm_{i}", nn.BatchNorm1d(fm_size//2))
            if self.negative_slope ==0.0:
                self.linear.add_module(f"relu_{i}", nn.ReLU())
            else:
                self.linear.add_module(f"lrelu_{i}", nn.LeakyReLU(self.negative_slope))
            if self.dropout != 0.0:
                self.linear.add_module(f"dropout_{i}", nn.Dropout(self.dropout))
            fm_size = fm_size//2
            i+=1
 
        self.linear.add_module(f"linear_{i}", nn.Linear(fm_size, 5))

    
    def forward(self, x):
        x = self.linear(x) 
        return x