import sys
sys.path.append("..")

import torch
import dacite
from torch import nn
from typing import Mapping, Any, List, Tuple

from modules import encoder, decoder
from dataclasses import dataclass
from modules.massagers import SimpleMassager, NoMassager


REGISTER = {
    "BMSHJEncoder": encoder.BMSHJEncoder,
    "BMSHJDecoder": decoder.BMSHJDecoder,
    "ConvEncoder": encoder.ConvEncoder,
    "ConvDecoder": decoder.ConvDecoder,
    "SimpleMassager": SimpleMassager,
    "NoMassager": NoMassager,
}


def build_registered_class(config: List[Any]):
    if config[0] not in REGISTER:
        return ValueError("Class to build has not been registered!")
    return REGISTER[config[0]](**config[1])


@dataclass 
class EncoderConfig:
    in_channels: int  = 3
    hidden_channels: int = 192


@dataclass
class DecoderConfig:
    out_channels: int = 3
    hidden_channels: int = 192


@dataclass
class MassagerConfig:
    channels: int  = 192


class GaussianVAE(nn.Module):
    def __init__(
        self,
        encoder_config: Mapping[str, Any],
        decoder_config: Mapping[str, Any],
        massager_config: Mapping[str, Any],
    ):
        super().__init__()

        self.encoder_config = dacite.from_dict(EncoderConfig, encoder_config[1])
        self.encoder = build_registered_class(encoder_config)
        self.decoder = build_registered_class(decoder_config)
        self.massager = build_registered_class(massager_config)

        self.post_conv = nn.Conv2d(
            in_channels=self.encoder_config.hidden_channels,
            out_channels=self.encoder_config.hidden_channels * 2,
            kernel_size=1,
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = torch.chunk(self.post_conv(self.encoder(x)), chunks=2, dim=1)
        return mean, logvar

    def massage(self, x: torch.Tensor) -> torch.Tensor:
        return self.massager(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO Take a look at this clamping 
        # https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/distributions/distributions.py#L24
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)

        return mean + std * torch.randn_like(mean)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        mean, logvar = self.encode(x)
        sample = self.sample(mean, logvar)
        massaged_sample = self.massager(sample)
        recon = self.decode(massaged_sample)
        return mean, logvar, recon
        