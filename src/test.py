import hydra
import time 
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from omegaconf import DictConfig, OmegaConf
from __init__ import configs_dir
from utils.utils import set_seed
from dataset.fmnist import FMNIST, Downstream_FMNIST
from model.model import Model, Downstream_Model
from utils.defaults import build_config

@hydra.main(version_base=None, config_path=configs_dir(), config_name="config")
def run_testing(cfg: DictConfig) -> None:

    config = build_config(cfg)

    set_seed(cfg.common.seed)

    tr_ep_loss = []
    tr_ep_acc = []
    val_ep_loss = []
    val_ep_acc = []


    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((config.DATA.mean,), (config.DATA.std,))])

    train_dataset = datasets.FashionMNIST('../data/FMNIST', download=True, train=True, transform=transform)
    val_dataset = datasets.FashionMNIST('../data/FMNIST', download=True, train=False, transform=transform)

    train_data = Downstream_FMNIST(cfg=config, phase='train', img=train_dataset, num_classes=config.MODEL.num_classes)
    train_loader = DataLoader(train_data,batch_size = cfg.common.batch_size, drop_last = True)

    val_data = Downstream_FMNIST(cfg=config, phase='valid', img=val_dataset,num_classes=config.MODEL.num_classes)
    val_loader = DataLoader(val_data,batch_size = cfg.common.batch_size, drop_last = True)

    model = Model(cfg=config, 
                  base_model=config.MODEL.base_model,
                  base_out_layer=config.MODEL.base_out_layer).to(cfg.common.device)
    downstream_model = Downstream_Model(cfg=config, 
                                        premodel=model, 
                                        num_classes=config.MODEL.num_classes).to(cfg.common.device)

    optimizer = torch.optim.SGD([params for params in downstream_model.parameters() if params.requires_grad],
                                lr = config.MODEL.test_lr, 
                                momentum = config.MODEL.momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                   step_size=1, 
                                                   gamma=config.MODEL.gamma, 
                                                   last_epoch=config.MODEL.last_epoch, 
                                                   verbose = True)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.common.test_epochs):

        stime = time.time()
        print("=============== Epoch : %3d ==============="%(epoch+1))

        loss_sublist = np.array([])
        acc_sublist = np.array([])

        #iter_num = 0
        downstream_model.train()

        optimizer.zero_grad()

        for x, y in train_loader:
            x = x.to(device = cfg.common.device, dtype = torch.float).repeat(1, 3, 1, 1)
            y = y.to(device = cfg.common.device)

            z = downstream_model(x)

            optimizer.zero_grad()

            tr_loss = loss_fn(z,y)
            tr_loss.backward()

            preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data))

            optimizer.step()

            loss_sublist = np.append(loss_sublist, tr_loss.cpu().data)
            acc_sublist = np.append(acc_sublist,np.array(np.argmax(preds,axis=1)==y.cpu().data.view(-1)).astype('int'),axis=0)

        print('ESTIMATING TRAINING METRICS.............')

        print('TRAINING BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist)*config.MODEL.accumulation_steps)
        print('TRAINING BINARY ACCURACY: ',np.mean(acc_sublist))

        tr_ep_loss.append(np.mean(loss_sublist))
        tr_ep_acc.append(np.mean(acc_sublist))

        print('ESTIMATING VALIDATION METRICS.............')

        downstream_model.eval()

        loss_sublist = np.array([])
        acc_sublist = np.array([])

        with torch.no_grad():
            for x,y in val_loader:
                x = x.to(device = cfg.common.device, dtype = torch.float).repeat(1, 3, 1, 1)
                y = y.to(device = cfg.common.device)
                z = downstream_model(x)

                val_loss = loss_fn(z,y)

                preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data))

                loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
                acc_sublist = np.append(acc_sublist,np.array(np.argmax(preds,axis=1)==y.cpu().data.view(-1)).astype('int'),axis=0)


        print('VALIDATION BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
        print('VALIDATION BINARY ACCURACY: ',np.mean(acc_sublist))

        val_ep_loss.append(np.mean(loss_sublist))
        val_ep_acc.append(np.mean(acc_sublist))

        lr_scheduler.step()

        if np.mean(loss_sublist) <= config.MODEL.min_val_loss:
            config.MODEL.min_val_loss = np.mean(loss_sublist)
            print('Saving model...')
            torch.save({'model_state_dict': downstream_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                f'{config.DATA.current_dir}/save_models/FMNIST_rn50_p128_sgd0p01_decay0p98_all_lincls_300621.pt')

        print("Time Taken : %.2f minutes"%((time.time()-stime)/60.0))

    # Inference
    downstream_model.eval()

    loss_sublist = np.array([])
    acc_sublist = np.array([])

    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device = cfg.common.device, dtype = torch.float).repeat(1, 3, 1, 1)
            y = y.to(device = cfg.common.device)
            z = downstream_model(x)

            val_loss = loss_fn(z,y)

            preds = torch.exp(z.cpu().data)/torch.sum(torch.exp(z.cpu().data))

            loss_sublist = np.append(loss_sublist, val_loss.cpu().data)
            acc_sublist = np.append(acc_sublist,np.array(np.argmax(preds,axis=1)==y.cpu().data.view(-1)).astype('int'),axis=0)

    print('TEST BINARY CROSSENTROPY LOSS: ',np.mean(loss_sublist))
    print('TEST BINARY ACCURACY: ',np.mean(acc_sublist))

    # Plot
    plt.plot([t for t in tr_ep_acc])
    plt.plot([t for t in val_ep_acc])
    plt.legend(['train','valid'])
    plt.savefig(f'{config.DATA.current_dir}/plots/acc.png')

    plt.plot(tr_ep_loss)
    plt.plot(val_ep_loss)
    plt.legend(['train','valid'])
    plt.savefig(f'{config.DATA.current_dir}/plots/loss.png')


if __name__ == "__main__":
    run_testing()