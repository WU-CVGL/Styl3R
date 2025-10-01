from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .encoder_noposplat_style import EncoderNoPoSplatStyle, EncoderNoPoSplatStyleCfg
from .encoder_noposplat_token_style import EncoderNoPoSplatTokenStyle, EncoderNoPoSplatTokenStyleCfg
from .encoder_noposplat_multi_token_style import EncoderNoPoSplatMultiTokenStyle
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
    # "noposplat_style": (EncoderNoPoSplatStyle, None),
    "noposplat_token_style": (EncoderNoPoSplatTokenStyle, None),
    "noposplat_multi_token_style": (EncoderNoPoSplatMultiTokenStyle, None)
}

EncoderCfg = EncoderNoPoSplatCfg | EncoderNoPoSplatTokenStyleCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
