import torch.nn as nn
import torch
import torch.nn.functional as F
from fastai.basic_train import DatasetType
from typing import Callable
from torch.distributions import Bernoulli

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

class DropLinear(nn.Module):
    def __init__(self, in_features, out_features, dp=0.):
        """Wrapper of a linear layer with a dropout module"""
        super(DropLinear, self).__init__()

        self.dropout = CustomDropout(dp)
        self.W = nn.Linear(in_features=in_features,out_features=out_features)
        
        self._mask = None
    
    @classmethod
    def sample_mask(cls,p,n):
        """Returns the mask of the weights"""
        
        bn = Bernoulli(p)
        mask = bn.sample((n,1))
        
        return mask
    
    def sample(self):
        """Sample a mask from the dropout module and the weight matrix"""
        p = 1 - self.dropout.dp
        shape = self.W.weight.data.shape
        n = shape[0]
        
        mask = DropLinear.sample_mask(p,n)
        
        return mask
    
    def topk_sample(self,n:int,k:int,p:float):
        """Sample a mask where only the top k weights of the matrix are considered for the dropout.
        The other weights will be necessarily dropped.
        
        Args:
            n: Total number of weights
            k: Number of weights with the best magnitude to consider for the dropout
            p: Probability to keep a neuron
            
        Returns:
            A mask where only the top k neurons will be used to sample from
        """
        
        # We compute the indexes of the rows with highest norm
        norm = self.W.weight.data.norm(dim=1)
        idx = torch.topk(norm,k).indices
        
        # We create a mask with only zeros first
        mask = torch.zeros(n)
        
        # Then we eventually allow the top k neurons to be kept depending on p
        small_mask = DropLinear.sample_mask(p,k).view(-1)
        mask[idx] = small_mask
        
        return mask 
        
    def set_mask(self,mask):
        self._mask = mask
        
    def remove_mask(self):
        self._mask = None
    
    def forward(self, x):
        if torch.is_tensor(self._mask):
            # We fix the weights with respect to the mask
            weight = self.W.weight.data
            bias = self.W.bias.data
            
            # We normalize as during training by dividing by the probability to keep a neuron
            dp = self.dropout.dp
            p = 1 - dp
            
            mask = self._mask
            if len(mask.shape) == 1:
                mask = mask.view(-1,1)
                
            # We apply our mask to the weights 
            masked_weight = mask.to(weight.device) * weight
            z = F.linear(x,masked_weight,bias)
            
            # Because we manually mask our weights we need to divide by the probability to keep a neuron
            z = z / p
        else:
            # We keep the stochasticity here so we do as usual
            z = self.W(x)
            z = self.dropout(z)
        return z