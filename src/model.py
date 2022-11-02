import os
import math
from tqdm import trange, tqdm
from pathlib import Path
import torch
from torch.nn.parameter import Parameter
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoModel

from enum import Enum, auto

from src.re-parameterize import DiffWeight, DiffWeightFixmask
from src.utils import dict_to_device


class ModelState(Enum):
    FINETUNING = auto()
    DIFFPRUNING = auto()
    FIXMASK = auto()


class DiffNetwork(torch.nn.Module): 
    
    def __init__(self, num_labels, *args, **kwargs):       
        super().__init__()
        self._model_state = ModelState.FINETUNING
        self._num_labels = num_labels
        self._log_ratio = 0
        self._l0_norm_term = 0
        self._alpha_weights = []
        self._z_mask = []
        
        self.encoder = AutoModel.from_pretrained(*args, **kwargs)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)
        
        self._init_weights(self.classifier)
        self._init_weights(self.encoder)
        
        self.to(self.device)

    def device(self) :
        return next(self.encoder.parameters()).device

    def model_type(self) :
        return self.encoder.config.model_type
    

    def model_name(self) :
        return self.encoder.config._name_or_path
    

    def total_layers(self) :
        possible_keys = ["num_hidden_layers", "n_layer"]
        for k in possible_keys:
            if k in self.encoder.config.__dict__:
                return getattr(self.encoder.config, k) + 2 # +2 for embedding layer and last layer
        raise Exception("number of layers of pre trained model could not be determined")
    

    def _parametrized(self) :
        return (self._model_state == ModelState.DIFFPRUNING or self._model_state == ModelState.FIXMASK)
        

    def get_log_ratio(concrete_lower: float, concrete_upper: float) :
        # calculate regularization term in objective
        return 0 if (concrete_lower == 0) else math.log(-concrete_lower / concrete_upper)
    

    def get_l0_norm_term(alpha: torch.Tensor, log_ratio: float) 
        return torch.sigmoid(alpha - log_ratio).sum()

    
    def get_encoder_base_modules(self, return_names: bool = False):
        if self._parametrized:
            check_fn = lambda m: hasattr(m, "parametrizations")
        else:
            check_fn = lambda m: len(m._parameters)>0
        return [(n,m) if return_names else m for n,m in self.encoder.named_modules() if check_fn(m)]
          
