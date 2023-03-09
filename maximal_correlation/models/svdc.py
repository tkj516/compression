import sys
sys.path.append("..")

import torch
from torch import nn

from compressai.entropy_models import EntropyBottleneck
from compressai.models import CompressionModel
from typing import Any, List, Mapping, Tuple, Optional
from ml_collections import ConfigDict
from dataclasses import dataclass

from modules.encoder import MNISTEncoder
from modules.decoder import MNISTDecoder
from utils.class_builder import ClassBuilder


class NoAugmentation:
    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, x: torch.Tensor, **kwargs):
        del kwargs
        return x


@dataclass
class EncoderConfig:
    in_channels: int = 3
    hidden_channels: int = 192
    out_channels: int = 256

ENCODER_REGISTER = {
    "MNISTEncoder": MNISTEncoder,
}
encoder_builder = ClassBuilder(ENCODER_REGISTER, EncoderConfig)

@dataclass
class DecoderConfig:
    in_channels: int = 256
    hidden_channels: int = 256
    out_channels: int = 3

DECODER_REGISTER = {
    "MNISTDecoder": MNISTDecoder,
}
decoder_builder = ClassBuilder(DECODER_REGISTER, DecoderConfig)


class SVDC(CompressionModel):
    def __init__(
        self,
        feature_encoder_config: ConfigDict,
        encoder_config: ConfigDict,
        decoder_config: ConfigDict,
    ):
        super().__init__()

        self.feature_encoder, self.feature_encoder_config = encoder_builder.build(feature_encoder_config)
        self.encoder, self.encoder_config = encoder_builder.build(encoder_config)
        self.decoder, self.decoder_config = decoder_builder.build(decoder_config)
        self.conditioner = nn.Conv2d(self.feature_encoder_config.out_channels, 2 * self.encoder_config.hidden_channels, 1, 1)

        self.entropy_bottleneck = EntropyBottleneck(
            self.encoder_config.out_channels)
    
    def feature_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(x)

    def encode(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(x, c)

    def quantize(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        return y_hat, y_likelihoods

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, inputs: List[torch.Tensor], mask_percent: float) -> Mapping[str, torch.Tensor]:
        # 1. Augment the input input image and get two samples
        x, z, _ = inputs

        # 2. Encode the augmented sample
        f_z = self.feature_encode(z)

        # 3. Mask the encoded augmented samples
        mask = torch.ones(*f_z.shape)
        mask[..., : int(f_z.shape[-1] * (1 - mask_percent))] = 0.0
        f_z_m = f_z * mask.to(f_z.device)

        # 4. Encode and compress the image
        c = self.conditioner(f_z_m)
        w = self.encode(x, c)
        w_hat, w_likelihoods = self.entropy_bottleneck(w)

        # 5. Decode the image
        x_hat = self.decode(torch.concatenate([f_z_m, w_hat], dim=1))

        return {
            "x_hat": x_hat,
            "w_likelihoods": w_likelihoods,
        }

    # def compress(self, x: torch.Tensor, num_samples: int) -> Mapping[str, Any]:
    #     # 1. Augment the input input image and get multiple samples
    #     augmented = self.augment(x, num_samples)
    #     x_i = [sample.squeeze(0) for sample in torch.chunk(augmented, chunks=num_samples, dim=0)]

    #     # 2. Encode the augmented samples and compute the expected values
    #     f_z = self.encode(torch.concatenate(x_i, dim=0))
    #     f_z = sum(torch.chunk(f_z, chunks=num_samples, dim=0)) / num_samples
        
    #     f_z_strings = self.entropy_bottleneck.compress(f_z)
    #     return{
    #         "y_strings": [f_z_strings],
    #         "shape": f_z.size()[-2:],
    #     }

    # def decompress(self, strings: Any, shape: torch.Size) -> Mapping[str, torch.Tensor]:
    #     y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
    #     x_hat = self.decode(y_hat).clamp(0, 1)
    #     return {"x_hat": x_hat}
