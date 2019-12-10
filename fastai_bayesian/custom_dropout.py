import torch.nn as nn
import torch
from fastai.basic_train import DatasetType
from typing import Callable

class CustomDropout(nn.Module):
    """Custom Dropout made to constantly keep stochasticity unless manually deactivated"""

    def __init__(self, dp:float, activate_stochasticity=True):
        """Initialize the module with a dropout probability and a boolean switch for stochasticty

        Args:
            activate_stochasticity: A boolean, saying wether or not we use stochasticity
            dp: The probability to drop a neuron 
        """
        super().__init__()
        self.activate_stochasticity = activate_stochasticity

        self.dp = dp

    def forward(self, x):
        return nn.functional.dropout(x, self.dp, training=self.training or self.activate_stochasticity)

    def extra_repr(self):
        return f"dp={self.dp}, activate_stochasticity={self.activate_stochasticity}"


def switch_custom_dropout(m, activate_stochasticity:bool=True, verbose:bool=False):
    """Turn all Custom Dropouts training mode to true or false according to the variable activate_stochasticity"""
    for c in m.children():
        if isinstance(c, CustomDropout):
            print(f"Current active : {c.activate_stochasticity}")
            print(f"Switching to : {activate_stochasticity}")
            c.activate_stochasticity = activate_stochasticity
        else:
            switch_custom_dropout(c, activate_stochasticity=activate_stochasticity)

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
