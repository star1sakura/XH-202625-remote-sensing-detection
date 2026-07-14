from __future__ import annotations

from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.mksnet_v2 import (
    MKSBlock,
    MKSChannelAttention,
    MKSSpatialAttention,
    MKSStage,
)
from xh_detect.models.sph_yolo import NAMBlock, SwinPredictionBlock


def register_custom_modules() -> None:
    """Expose custom modules to Ultralytics YAML parsing and checkpoint loading."""
    import ultralytics.nn.tasks as tasks

    tasks.MKSNetLiteBlock = MKSNetLiteBlock
    tasks.MKSChannelAttention = MKSChannelAttention
    tasks.MKSSpatialAttention = MKSSpatialAttention
    tasks.MKSBlock = MKSBlock
    tasks.MKSStage = MKSStage
    tasks.NAMBlock = NAMBlock
    tasks.SwinPredictionBlock = SwinPredictionBlock
