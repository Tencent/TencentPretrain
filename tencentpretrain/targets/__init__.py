from tencentpretrain.targets.mlm_target import MlmTarget
from tencentpretrain.targets.sp_target import SpTarget
from tencentpretrain.targets.lm_target import LmTarget
from tencentpretrain.targets.cls_target import ClsTarget
from tencentpretrain.targets.bilm_target import BilmTarget
from tencentpretrain.targets.clr_target import ClrTarget
from tencentpretrain.targets.target import Target


str2target = {"sp": SpTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "cls": ClsTarget, "clr": ClrTarget}

__all__ = ["Target", "SpTarget", "MlmTarget", "LmTarget", "BilmTarget", "ClsTarget", "ClrTarget", "str2target"]
