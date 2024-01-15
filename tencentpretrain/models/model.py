import torch.nn as nn


class Model(nn.Module):
    """
    Pretraining models consist of three (five) parts:
        - embedding
        - encoder
        - tgt_embedding (optional)
        - decoder (optional)
        - target
    """

    def __init__(self, args, embedding, encoder, tgt_embedding, decoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.tgt_embedding = tgt_embedding
        self.decoder = decoder
        self.target = target

        if "mlm" in args.target and args.tie_weights:
            self.target.mlm.linear_2.weight = self.embedding.word.embedding.weight
        elif "lm" in args.target and args.tie_weights and self.tgt_embedding is not None and "word" in self.tgt_embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.tgt_embedding.word.embedding.weight
        elif "lm" in args.target and args.tie_weights and "word" in self.embedding.embedding_name_list:
            self.target.lm.output_layer.weight = self.embedding.word.embedding.weight
            
        if self.decoder is not None and args.share_embedding:
            self.tgt_embedding.word.embedding.weight = self.embedding.word.embedding.weight

        if args.freeze_parameters:
            name_mapping = {
                "embedding": self.embedding, "encoder": self.encoder, "tgt_embedding": self.tgt_embedding,
                "decoder": self.decoder, "target": self.target
                }
            for freeze_name in args.freeze_parameters:
                if name_mapping[freeze_name] is None:
                    continue
                for name, param in name_mapping[freeze_name].named_parameters():
                    if args.freeze_exclude_by_name == "" or args.freeze_exclude_by_name not in name:
                        param.requires_grad = False

    def forward(self, src, tgt, seg, tgt_in=None, tgt_seg=None):
        emb = self.embedding(src, seg)
        memory_bank = self.encoder(emb, seg)
        if self.decoder:
            tgt_emb = self.tgt_embedding(tgt_in, tgt_seg)
            memory_bank = self.decoder(memory_bank, tgt_emb, (seg, tgt_seg))

        if tgt_seg is not None:
            loss_info = self.target(memory_bank, tgt, tgt_seg)
        else:
            loss_info = self.target(memory_bank, tgt, seg)

        return loss_info
