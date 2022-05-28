from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
import torch

from pathlib import Path
import wandb

from datamodule import Datamodule
from params import (
    RANDOM_SEED,
    LocationConfig,
    TrainingConfig,
    WandbConfig,
    NetworkConfig,
)
from models.mlp import MLP, BiggerMLP


def get_all_checkpoints():
    checkpoints_dir = Path(LocationConfig.checkpoints_dir)
    list_of_checkpoints = list(checkpoints_dir.glob("*.ckpt"))

    return list_of_checkpoints


def init_MLP(lr, batch_norm, negative_slope, dropout, batch_size) -> MLP:
    mlp = MLP(
        lr=lr,
        batch_norm=batch_norm,
        negative_slope=negative_slope,
        dropout = dropout,
        batch_size = batch_size
        )
    return mlp

def init_BiggerMLP(lr, batch_norm, negative_slope, dropout, batch_size) -> BiggerMLP:
    biggerMLP = BiggerMLP(
        lr=lr,
        batch_norm=batch_norm,
        negative_slope=negative_slope,
        dropout = dropout,
        batch_size = batch_size
        )
    return biggerMLP


def save_model_from_last_checkpoint_as_state_dict() -> None:
    list_of_checkpoints = get_all_checkpoints()
    latest_checkpoint_path = max(list_of_checkpoints, key=lambda p: p.stat().st_ctime)

    lightning_model = init_MLP()
    lightning_model.load_from_checkpoint(latest_checkpoint_path)
    lightning_model.eval()
    lightning_model = lightning_model.cpu()

    best_model_path = Path(LocationConfig.best_model)
    torch.save(lightning_model.state_dict(), best_model_path)

    print("Saved the latest model at:", best_model_path)


def run_train(dm, model):

    chkp_dir = Path(LocationConfig.checkpoints_dir)
    modelCheckpoint = ModelCheckpoint(
        dirpath=chkp_dir,
        save_top_k=1,
        verbose=True,
        monitor="val_loss_epoch",
        mode="min",
    )

    earlyStopping = EarlyStopping(
        monitor="val_loss_epoch",
        patience=TrainingConfig.patience,
    )

    wandb_logger = WandbLogger(
        project=WandbConfig.project_name,
        save_dir=WandbConfig.save_dir,
        name=WandbConfig.run_name,
        entity=WandbConfig.entity,
    )

    trainer = Trainer(
        max_epochs=TrainingConfig.epochs,
        gpus=TrainingConfig.gpus,
        deterministic=TrainingConfig.deterministic,
        accumulate_grad_batches=TrainingConfig.accumulate_grad_batches,
        callbacks=[earlyStopping, modelCheckpoint],
        logger=wandb_logger,
    )

    # Train model
    trainer.fit(model, dm)

    # Save model
    save_model_from_last_checkpoint_as_state_dict()
    
    
def sweep_iteration():
    wandb.init()    # required to have access to `wandb.config`
    
    chkp_dir = Path(LocationConfig.checkpoints_dir)
    modelCheckpoint = ModelCheckpoint(
        dirpath=chkp_dir,
        save_top_k=1,
        verbose=True,
        monitor="val_acc_epoch",
        mode="max",
    )

    earlyStopping = EarlyStopping(
        monitor="val_acc_epoch",
        mode="max",
        patience=wandb.config.batch_size,
    )
    
    # set up W&B logger
    
    
    run_name = str(NetworkConfig.negative_slope) + '_'
    run_name += str(NetworkConfig.dropout)
    wandb_logger = WandbLogger(
        project=WandbConfig.project_name,
        save_dir=WandbConfig.save_dir,
        name=run_name,
        entity=WandbConfig.entity,
    )
    
    train_data_path = Path(LocationConfig.enc + 'train/')
    test_data_path = Path(LocationConfig.enc + 'test/')
    dm = Datamodule(
        batch_size=wandb.config.batch_size,
        train_dir=train_data_path,
        val_dir=test_data_path,
    )
    
    # setup model - note how we refer to sweep parameters with wandb.config
    model = init_MLP(
        negative_slope=wandb.config.negative_slope, 
        batch_size = wandb.config.batch_size,
        batch_norm=wandb.config.batch_norm, 
        dropout=wandb.config.dropout,
        lr=wandb.config.lr, 
    )
    
    # setup Trainer
    trainer = Trainer(
        accumulate_grad_batches=TrainingConfig.accumulate_grad_batches,
        deterministic=TrainingConfig.deterministic,
        callbacks=[earlyStopping, modelCheckpoint],
        max_epochs=wandb.config.batch_size*10,
        gpus=TrainingConfig.gpus,
        logger=wandb_logger,
    )
    
    # train
    trainer.fit(model, dm)
    
       
def init_output_dirs() -> None:
    Path(LocationConfig.checkpoints_dir).mkdir(exist_ok=True, parents=True)
    Path(LocationConfig.best_model).parent.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    init_output_dirs()
    pl.seed_everything(RANDOM_SEED)

    # train_data_path = Path(LocationConfig.new_data + 'train/')
    # test_data_path = Path(LocationConfig.new_data + 'test/')
    # dm = Datamodule(
    #     batch_size=TrainingConfig.batch_size,
    #     train_dir=train_data_path,
    #     val_dir=test_data_path,
    # )
    # model = init_BiggeMLP(
    #     lr=TrainingConfig.lr, 
    #     batch_norm=NetworkConfig.batch_norm, 
    #     negative_slope=NetworkConfig.negative_slope, 
    #     dropout=NetworkConfig.dropout, 
    #     batch_size=TrainingConfig.batch_size
    # )
    # run_train(dm, model)
    
    sweep_config = {
      "method": "random",
      "metric": {
          "name": "val_acc_epoch",
          "goal": "maximize"
      },
#       "parameters": {
#             "batch_norm": {"values": [True, False]}, 
#             "dropout": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]}, 
#             "negative_slope": {"values": [0.0, 0.01, 0.02, 0.05, 0.1]},
#             "lr": {"values": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]},
#             "batch_size": {"values": [2, 4, 8, 16, 32, 64, 128]},
#         },
      "parameters": {
            "batch_norm": {"values": [False, True]}, 
            "dropout": {"values": [0.0, 0.1, 0.2, 0.3, 0.4]}, 
            "negative_slope": {"values": [0.0, 0.01, 0.02, 0.05, 0.1]},
            "lr": {"values": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 5e-3, 5e-4, 5e-5, 5e-6, 5e-7]},
            "batch_size": {"values": [2, 4, 8, 16, 32, 64, 128, 256]},
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project=WandbConfig.project_name)
    wandb.agent(sweep_id, function=sweep_iteration, count=13)