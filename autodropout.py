
import torch
import torch.nn as nn


class PLU(nn.Module):
    """Probability Linear Unit"""

    def __init__(self):
        super(PLU, self).__init__()

    def forward(self, x):
        z = torch.clamp(x, 0, 1)
        return z


class CustomDropout(nn.Module):
    """Custom Dropout module to be used as a baseline for MC Dropout"""
  def __init__(self,p,activate=True):
    super().__init__()
    self.activate = activate
    self.p = p
    
  def forward(self,x):
    return nn.functional.dropout(x,self.p,training=self.activate)

class AutoDropout(nn.Module):
    def __init__(self, dp=0., requires_grad=False):

        super(AutoDropout, self).__init__()

        # We transform the dropout rate to keep rate
        p = 1 - dp
        p = torch.tensor(p)

        self.plu = PLU()

        if requires_grad:
            p = nn.Parameter(p)
            self.register_parameter("p", p)
        else:
            self.register_buffer("p", p)

    def forward(self, x):
        bs, shape = x.shape[0], x.shape[1:]

        # We make sure p is a probability
        p = self.plu(self.p)

        # We sample a mask
        m = Bernoulli(p).sample(shape)

        # Element wise multiplication
        z = x * m

        return z


class DropLinear(nn.Module):
    def __init__(self, in_features, out_features, dp=0., bias=True, requires_grad=False):
        super(DropLinear, self).__init__()

        self.dp = AutoDropout(dp=dp, requires_grad=requires_grad)
        self.W = nn.Linear(in_features=in_features,
                           out_features=out_features, bias=bias)
        self.W.weight.data = self.W.weight.data / self.W.weight.data.norm() * (1-dp)

    def forward(self, x):
        z = self.W(x)
        z = self.dp(z)
        return z
