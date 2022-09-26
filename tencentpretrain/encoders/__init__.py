from tencentpretrain.encoders.transformer_encoder import TransformerEncoder
from tencentpretrain.encoders.rnn_encoder import RnnEncoder
from tencentpretrain.encoders.rnn_encoder import LstmEncoder
from tencentpretrain.encoders.rnn_encoder import GruEncoder
from tencentpretrain.encoders.rnn_encoder import BirnnEncoder
from tencentpretrain.encoders.rnn_encoder import BilstmEncoder
from tencentpretrain.encoders.rnn_encoder import BigruEncoder
from tencentpretrain.encoders.cnn_encoder import GatedcnnEncoder
from tencentpretrain.encoders.dual_encoder import DualEncoder


str2encoder = {"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder,
               "gatedcnn": GatedcnnEncoder, "dual": DualEncoder}

__all__ = ["TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "GatedcnnEncoder", "DualEncoder", "str2encoder"]
