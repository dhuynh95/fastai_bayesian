import torch.nn as nn
import torch
from fastai.basic_train import DatasetType
from typing import Callable

class CustomDropout(nn.Module):
    """Custom Dropout module to be used as a baseline for MC Dropout"""

    def __init__(self, p:float, activate=True):
        super().__init__()
        self.activate = activate
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, self.p, training=self.training or self.activate)

    def extra_repr(self):
        return f"p={self.p}, activate={self.activate}"


def switch_custom_dropout(m, activate:bool=True, verbose:bool=False):
    """Turn all Custom Dropouts training mode to true or false according to the variable activate"""
    for c in m.children():
        if isinstance(c, CustomDropout):
            print(f"Current active : {c.activate}")
            print(f"Switching to : {activate}")
            c.activate = activate
        else:
            switch_custom_dropout(c, activate=activate)

def convert_layers(model:nn.Module, original:nn.Module, replacement:nn.Module, get_args:Callable=None,
 additional_args:dict={}):
    """Convert modules of type "original" to "replacement" inside the model
    
    get_args : a function to use on the original module to eventually get its arguements to pass to the new module
    additional_args : a dictionary to add more args to the new module

    """
    for child_name, child in model.named_children():

        if isinstance(child, original):
            # First we grab args from the child
            if get_args:
                original_args = get_args(child)
            else:
                original_args = {}

            # If we want to provide additional args
            if additional_args:
                args = {**original_args, **additional_args}
            else:
                args = original_args

            new_layer = replacement(**args)
            setattr(model, child_name, new_layer)
        else:
            convert_layers(child, original, replacement,
                           get_args, additional_args)
