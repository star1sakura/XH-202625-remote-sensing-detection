from __future__ import annotations

from xh_detect.models.mksnet_lite import MKSNetLiteBlock


def register_custom_modules() -> None:
    """Expose custom modules to Ultralytics YAML parsing and checkpoint loading."""
    import ultralytics.nn.tasks as tasks

    tasks.MKSNetLiteBlock = MKSNetLiteBlock
