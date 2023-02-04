import sys
sys.path.append("..")

import dacite 
import torch

from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from typing import Any, Mapping
from ml_collections import ConfigDict
from dataclasses import dataclass

from modules.additive_gaussian_noise import AdditiveGaussianNoise
from modules.encoder import BMSHJEncoder, ELICEncoder
from modules.decoder import BMSHJDecoder, ELICDecoder
from utils.class_builder import ClassBuilder


class NoAugmentation:
    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, x: torch.Tensor, num_samples: int, **kwargs):
        del kwargs
        return x.unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)


AUGMENTATION_REGISTER = {
    "AdditiveGaussianNoise": AdditiveGaussianNoise,
    "NoAugmentation": NoAugmentation,
}
augmentation_builder = ClassBuilder(AUGMENTATION_REGISTER)


@dataclass
class EncoderConfig:
    in_channels: int = 3
    hidden_channels: int = 192
    out_channels: int = 256

ENCODER_REGISTER = {
    "BMSHJEncoder": BMSHJEncoder,
    "ELICEncoder": ELICEncoder
}
encoder_builder = ClassBuilder(ENCODER_REGISTER, EncoderConfig)

@dataclass
class DecoderConfig:
    in_channels: int = 256
    hidden_channels: int = 256
    out_channels: int = 3

DECODER_REGISTER = {
    "BMSHJDecoder": BMSHJDecoder,
    "ELICDecoder": ELICDecoder
}
decoder_builder = ClassBuilder(DECODER_REGISTER, DecoderConfig)


class SVDC(CompressionModel):
    def __init__(
        self,
        augmentation_config: ConfigDict,
        encoder_config: ConfigDict,
        decoder_config: ConfigDict,
    ):
        super().__init__()

        self.augmentation = augmentation_builder.build_class(augmentation_config)
        self.encoder, self.encoder_config = encoder_builder.build(encoder_config)
        self.decoder, self.decoder_config = decoder_builder.build(decoder_config)
        assert self.encoder_config.out_channels == self.decoder_config.in_channels

        self.entropy_bottleneck = EntropyBottleneck(
            self.encoder_config.out_channels)

    def augment(self, x: torch.Tensor, num_samples: int) -> torch.Tensor:
        return self.augmentation(x, num_samples)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor, num_samples: int) -> Mapping[str, torch.Tensor]:
        x_a = self.augment(x, num_samples).mean(0)
        y = self.encode(x_a)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return {
            "x_hat": x_hat,
            "y_likelihoods": y_likelihoods,
        }

    def compress(self, x: torch.Tensor, num_samples: int) -> Mapping[str, Any]:
        x_a = self.augment(x, num_samples)
        y = self.encode(x_a)
        y_strings = self.entropy_bottleneck.compress(y)
        return {
            "y_strings": [y_strings],
            "shape": y.size()[-2:],
        }

    def decompress(self, strings: Any, shape: torch.Size) -> Mapping[str, torch.Tensor]:
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.decode(y_hat).clamp(0, 1)
        return {"x_hat": x_hat}
