import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig


@dataclass
class DataConfig:
    current_dir: str
    
    HorizontalFlip: float = 0.5
    ResizedCrop: int = 32
    Cropheight: float = 0.8
    Cropwidth: float = 1.0

    mean: float = 0.5
    std: float = 0.5

@dataclass
class Model:
    base_model: str = "resnet50"
    base_out_layer: str = "avgpool"
    train_lr: float = 0.2
    weight_decay: float = 1e-6
    T_0: int = 500
    eta_min: float = 0.05
    last_epoch: int = -1
    temperature: float = 0.5
    
    # projectionhead_info
    in_features: int = 2048
    hidden_features: int = 2048
    out_features: int = 128

    num_classes: int = 10
    test_lr: float = 0.01
    momentum: float = 0.9
    gamma: float = 0.98
    min_val_loss: float = 100.0
    accumulation_steps: int = 1
    

@dataclass
class Config:
    DATA: DataConfig
    MODEL: Model


def build_config(cfg: DictConfig) -> None:
    current_dir = os.path.expanduser(cfg['dirs']['current_dir'])

    DATA = DataConfig(current_dir=current_dir,)
    
    MODEL = Model()

    config = Config(DATA=DATA,
                    MODEL=MODEL)
    return config