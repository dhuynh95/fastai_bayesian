
from fastai.callbacks.hooks import HookCallback
from fastai.torch_core import to_np

from autodropout import DropLinear
import pandas as pd
import torch
import torch.nn as nn


def norm2(x): return (x**2).sum()


def neg_entropy(p): return p * torch.log(p) + (1-p) * torch.log(1-p)


def get_layer(m, buffer, layer):
    """Function which takes a list and a model append the elements"""
    for c in m.children():
        if isinstance(c, layer):
            if isinstance(buffer, list):
                buffer.append(c)
            elif isinstance(buffer, dict):
                i = hex(id(c))
                buffer[i] = c
        get_layer(c, buffer, layer)


class CustomActivationStats(HookCallback):
    def __init__(self, learn, layer_type, do_remove: bool = True):
        super().__init__(learn)

        buffer = []
        get_layer(learn.model, buffer, layer_type)
        if not buffer:
            raise NotImplementedError(f"No {layer_type} Linear found")

        self.modules = buffer
        self.do_remove = do_remove

        self.stats = []

    def hook(self, m, i, o):
        mean = o.mean().item()
        std = o.std().item()
        z = to_np(o.mean(dim=0))

        i = hex(id(m))
        self.stats.append({"m": mean, "s": std, "z": z, "module": i})

    @property
    def df(self): return pd.DataFrame(self.stats)


class KLHook(HookCallback):
    """Hook to register the parameters of the latents during the forward pass to compute the KL term of the VAE"""

    def __init__(self, learn, do_remove: bool = True, recording=False):
        super().__init__(learn)

        # First we store all the DropLinears layers to hook them
        buffer = []
        get_layer(learn.model, buffer, DropLinear)
        if not buffer:
            raise NotImplementedError(f"No {DropLinear} Linear found")

        self.modules = buffer
        self.do_remove = do_remove

        # We will store the KL of each DropLinear here before summing them
        self.kls = []

        self.recording = recording

        if recording:
            self.stats = []
            self.loss = []

    def on_backward_begin(self, last_loss, **kwargs):

        total_kl = torch.tensor(self.kls).sum()
        total_loss = last_loss + total_kl

        if self.recording:
            self.loss.append({"total_kl": total_kl.item(), "last_loss": last_loss.item(),
                              "total_loss": total_loss.item()})

        # We empty the buffer of kls
        self.kls = []

        return {"last_loss": total_loss}

    def hook(self, m: nn.Module, i, o):
        "Save the latents of the bottleneck"
        p = m.dp.p
        p = m.dp.plu(p)

        ne = neg_entropy(p)

        W = m.W.weight
        norm_w = norm2(W)

        b = m.W.bias
        norm_b = norm2(b)

        kl = p * norm_w + ne + norm_b

        self.kls.append(kl)

        if self.recording:
            i = hex(id(m))
            self.stats.append(
                {"dropout": 1 - p.item(), "w": norm_w.item(), "ne": ne.item(), "module": i})

    @property
    def df_stats(self): return pd.DataFrame(self.stats)

    @property
    def df_loss(self): return pd.DataFrame(self.loss)

    def plot_stats(self, module=None):
        assert self.recording, "Recording mode was off during initialization"
        df = self.df_stats
        if module:
            df = df.loc[df.module == module]

        fig, ax = plt.subplots(3, 1, figsize=(16, 12))

        ax[0].plot(df.w.values)
        ax[0].set_title("Weight norm")

        ax[1].plot(df.dropout.values)
        ax[1].set_title("Dropout rate")

        ax[2].plot(df["ne"].values)
        ax[2].set_title("Negative entropy")

    def plot_losses(self):
        assert self.recording, "Recording mode was off during initialization"
        df = self.df_loss
        df.plot()
