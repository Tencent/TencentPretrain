from tencentpretrain.targets.mlm_target import MlmTarget
from tencentpretrain.targets.sp_target import SpTarget
from tencentpretrain.targets.lm_target import LmTarget
from tencentpretrain.targets.lmdecoder_target import LmDecoderTarget
from tencentpretrain.targets.cls_target import ClsTarget
from tencentpretrain.targets.ner_target import NerTarget
from tencentpretrain.targets.bilm_target import BilmTarget
from tencentpretrain.targets.clr_target import ClrTarget
from tencentpretrain.targets.target import Target


str2target = {"sp": SpTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "cls": ClsTarget, "clr": ClrTarget, "ner":NerTarget, "lmdecoder":LmDecoderTarget}

__all__ = ["Target", "SpTarget", "MlmTarget", "LmTarget", "BilmTarget", "ClsTarget", "NerTarget", "ClrTarget", "str2target", "LmDecoderTarget"]
