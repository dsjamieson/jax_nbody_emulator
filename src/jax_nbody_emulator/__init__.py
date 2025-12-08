"""
N-Body Emulator: A JAX-based neural network emulator for cosmological simulations.
"""

from .cosmology import D, H, f, dlogH_dloga, vel_norm, acc_norm
from .style_layers import (
    StyleBase3D,
    StyleConv3D,
    StyleSkip3D,
    StyleDownSample3D,
    StyleUpSample3D,
    LeakyReLUStyled,
)
from .style_blocks import (
    StyleResampleBlock3D,
    StyleResNetBlock3D,
)
from .nbody_emulator import NBodyEmulator
from .style_layers_vel import (
    StyleBase3DVel,
    StyleConv3DVel,
    StyleSkip3DVel,
    StyleDownSample3DVel,
    StyleUpSample3DVel,
    LeakyReLUStyledVel,
)
from .style_blocks_vel import (
    StyleResampleBlock3DVel,
    StyleResNetBlock3DVel,
)
from .nbody_emulator_vel import NBodyEmulatorVel
from .subbox import SubboxConfig, SubboxProcessor, SubboxProcessorVel


__version__ = "0.1.0"
__author__ = "Drew Jamieson"
__email__ = "drew.s.jamieson@gmail.com"

__all__ = [
    # Cosmology functions
    "D",
    "H", 
    "f",
    "dlogH_dloga",
    "growth_acc",
    "vel_norm",
    "acc_norm",
    # Style layers
    "StyleBase3D",
    "StyleConv3D",
    "StyleSkip3D", 
    "StyleDownSample3D",
    "StyleUpSample3D",
    "LeakyReLUStyled",
    # Style blocks
    "StyleResampleBlock3D",
    "StyleResNetBlock3D",
    # Main model
    "NBodyEmulator",
    # Style layers with velocity
    "StyleBase3DVel",
    "StyleConv3DVel",
    "StyleSkip3DVel", 
    "StyleDownSample3DVel",
    "StyleUpSample3DVel",
    "LeakyReLUStyledVel",
    # Style blocks
    "StyleResampleBlock3DVel",
    "StyleResNetBlock3DVel",
    # Main model
    "NBodyEmulatorVel",
    # Subbox processors
    "SubboxConfig",
    "SubboxProcessor",
    "SubboxProcessorVel"
]
