import time
import hydra
from omegaconf import DictConfig, OmegaConf
from __init__ import configs_dir

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from dataset.fmnist import FMNIST
from model.model import Model
from utils.loss import SimCLR_Loss
from utils.optimizer import LARS
from utils.fn import train_fn, valid_fn
from utils.utils import save_model, set_seed
from utils.defaults import build_config


@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_training(cfg: DictConfig) -> None:

    config = build_config(cfg)

    nr = 0
    current_epoch = 0
    tr_loss = []
    val_loss = []

    set_seed(cfg.common.seed)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((config.DATA.mean,), (config.DATA.std,))])

    train_dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    val_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

    train_data = FMNIST(cfg=config, phase='train', img=train_dataset)
    train_loader = DataLoader(train_data, batch_size = cfg.common.batch_size, drop_last=True)

    val_data = FMNIST(cfg=config, phase='valid', img=val_dataset)
    val_loader = DataLoader(val_data,batch_size = cfg.common.batch_size,drop_last=True)

    model = Model(cfg=config, 
                  base_model=config.MODEL.base_model,
                  base_out_layer=config.MODEL.base_out_layer)
    model = model.to(cfg.common.device)

    optimizer = LARS(
        [params for params in model.parameters() if params.requires_grad],
        lr=config.MODEL.train_lr,
        weight_decay=config.MODEL.weight_decay,
        exclude_from_weight_decay=["batch_normalization", "bias"],
    )

    # decay the learning rate with the cosine decay schedule without restarts
    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : (epoch+1)/10.0, verbose = True)
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.MODEL.T_0, 
                                                                         eta_min=config.MODEL.eta_min, 
                                                                         last_epoch=config.MODEL.last_epoch, verbose = True)

    criterion = SimCLR_Loss(batch_size = cfg.common.batch_size, temperature = config.MODEL.temperature)

    for epoch in range(cfg.common.train_epochs):
            
        print(f"Epoch [{epoch}/{cfg.common.train_epochs}]\t")
        stime = time.time()

        model.train()
        tr_loss_epoch = train_fn(train_loader, model, criterion, optimizer)

        if nr == 0 and epoch < 10:
            warmupscheduler.step()
        if nr == 0 and epoch >= 10:
            mainscheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]

        if nr == 0 and (epoch+1) % 50 == 0:
            save_model(cfg, model, optimizer, mainscheduler, current_epoch,
                       "SimCLR_FMNIST_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")

        model.eval()
        with torch.no_grad():
            val_loss_epoch = valid_fn(val_loader, model, criterion)

        if nr == 0:
            
            tr_loss.append(tr_loss_epoch / len(train_loader))
            
            val_loss.append(val_loss_epoch / len(val_loader))
            
            print(
                f"Epoch [{epoch}/{cfg.common.train_epochs}]\t Training Loss: {tr_loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
            )
            print(
                f"Epoch [{epoch}/{cfg.common.train_epochs}]\t Validation Loss: {val_loss_epoch / len(val_loader)}\t lr: {round(lr, 5)}"
            )
            current_epoch += 1

        time_taken = (time.time()-stime)/60
        print(f"Epoch [{epoch}/{cfg.common.train_epochs}]\t Time Taken: {time_taken} minutes")

    ## end training
    save_model(cfg=config, 
               model=model, 
               optimizer=optimizer, 
               scheduler=mainscheduler, 
               current_epoch=current_epoch, 
               name="SimCLR_FMNIST_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")

    plt.plot(tr_loss,'b-')
    plt.plot(val_loss,'r-')
    plt.legend(['t','v'])
    plt.savefig(f'{config.DATA.current_dir}/plots/training_result.png')

if __name__ == "__main__":
    run_training()