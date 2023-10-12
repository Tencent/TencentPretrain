from tencentpretrain.utils.dataset import *
from tencentpretrain.utils.dataloader import *
from tencentpretrain.utils.act_fun import *
from tencentpretrain.utils.optimizers import *
from tencentpretrain.utils.adversarial import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer,
                 "bpe": BPETokenizer, "xlmroberta": XLMRobertaTokenizer, "image": ImageTokenizer,
                 "text_image": TextImageTokenizer, "virtual": VirtualTokenizer, "hfpretrained": HFPreTrainedTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset, "mt": MtDataset,
               "t5": T5Dataset, "gsg": GsgDataset, "bart": BartDataset,
               "cls": ClsDataset, "prefixlm": PrefixlmDataset, "cls_mlm": ClsMlmDataset,
               "vit": VitDataset, "vilt": ViltDataset, "clip": ClipDataset, "s2t": S2tDataset,
               "beit":BeitDataset, "dalle": DalleDataset, "alpaca": AlpacaDataset}
str2dataloader = {"bert": BertDataloader, "lm": LmDataloader, "mlm": MlmDataloader,
                  "bilm": BilmDataloader, "albert": AlbertDataloader, "mt": MtDataloader,
                  "t5": T5Dataloader, "gsg": GsgDataloader, "bart": BartDataloader,
                  "cls": ClsDataloader, "prefixlm": PrefixlmDataloader, "cls_mlm": ClsMlmDataloader,
                  "vit": VitDataloader, "vilt": ViltDataloader, "clip": ClipDataloader, "s2t": S2tDataloader,
                  "beit":BeitDataloader, "dalle": DalleDataloader, "alpaca": AlpacaDataloader}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}

str2scheduler = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup,
                 "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                 "polynomial": get_polynomial_decay_schedule_with_warmup,
                 "constant": get_constant_schedule, "constant_with_warmup": get_constant_schedule_with_warmup,
                 "inverse_sqrt": get_inverse_square_root_schedule_with_warmup, "tri_stage": get_tri_stage_schedule}

str2adv = {"fgm": FGM, "pgd": PGD}

__all__ = ["CharTokenizer", "SpaceTokenizer", "BertTokenizer", "BPETokenizer", "XLMRobertaTokenizer",
           "ImageTokenizer", "TextImageTokenizer", "str2tokenizer",
           "BertDataset", "LmDataset", "MlmDataset", "BilmDataset",
           "AlbertDataset", "MtDataset", "T5Dataset", "GsgDataset",
           "BartDataset", "ClsDataset", "PrefixlmDataset", "ClsMlmDataset",
           "VitDataset", "ViltDataset", "ClipDataset", "BeitDataset", "DalleDataset", "AlpacaDataset", "str2dataset",
           "BertDataloader", "LmDataloader", "MlmDataloader", "BilmDataloader",
           "AlbertDataloader", "MtDataloader", "T5Dataloader", "GsgDataloader",
           "BartDataloader", "ClsDataloader", "PrefixlmDataloader", "ClsMlmDataloader",
           "VitDataloader", "ViltDataloader", "ClipDataloader", "BeitDataloader", "DalleDataloader","AlpacaDataloader","str2dataloader",
           "gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
           "AdamW", "Adafactor", "str2optimizer",
           "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
           "get_cosine_with_hard_restarts_schedule_with_warmup",
           "get_polynomial_decay_schedule_with_warmup",
           "get_constant_schedule", "get_constant_schedule_with_warmup", 
           "get_inverse_square_root_schedule_with_warmup", "get_tri_stage_schedule", "str2scheduler",
           "FGM", "PGD", "str2adv"]
