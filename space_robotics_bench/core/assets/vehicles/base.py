from typing import Optional, Union

from space_robotics_bench.core.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    AssetCfg,
    FrameCfg,
    RigidObjectCfg,
)


class VehicleCfg(AssetCfg):
    ## Model
    asset_cfg: Union[AssetBaseCfg, ArticulationCfg, RigidObjectCfg]

    ## Frames
    frame_manipulator_base: FrameCfg
    frame_camera_base: Optional[FrameCfg]
    frame_cargo_bay: FrameCfg
