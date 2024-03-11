from argparse import Namespace
import torch
import torch.nn as nn
import copy

from tencentpretrain.embeddings.embedding import Embedding
from tencentpretrain.embeddings.word_embedding import WordEmbedding
from tencentpretrain.embeddings.pos_embedding import PosEmbedding
from tencentpretrain.embeddings.patch_embedding import PatchEmbedding
from tencentpretrain.encoders import str2encoder,TransformerEncoder

str2embedding = {"word": WordEmbedding, "pos": PosEmbedding, "patch": PatchEmbedding}


class VisionLanguageEmbedding(nn.Module):
    '''
    an combination of a vision encoder and a text embedding
    '''
    def __init__(self, args, vocab_size):
        super(VisionLanguageEmbedding, self).__init__()
        # vision model for vision features
        vision_encoder_args = copy.deepcopy(vars(args))
        vision_encoder_args.update(args.vision_language_emb["vision_encoder"])
        vision_encoder_args = Namespace(**vision_encoder_args)
        self.vision_embedding = Embedding(vision_encoder_args)
        for embedding_name in vision_encoder_args.embedding:
            tmp_emb = str2embedding[embedding_name](vision_encoder_args, None)
            self.vision_embedding.update(tmp_emb, embedding_name)
        self.vision_encoder = str2encoder[vision_encoder_args.encoder](vision_encoder_args)

        # map the output of vision model into the same space as the text features
        projection_args = copy.deepcopy(vars(args))
        projection_args.update(args.vision_language_emb["projection"])
        projection_args = Namespace(**projection_args)
        projection_modules = [nn.Linear(vision_encoder_args.emb_size, projection_args.mlp_hidden_size)]
        for _ in range(1, projection_args.num_mlp_layer):
            projection_modules.append(nn.GELU())
            projection_modules.append(nn.Linear(projection_args.mlp_hidden_size, projection_args.mlp_hidden_size))
        self.projection = nn.Sequential(*projection_modules)

        # text embedding
        text_args = copy.deepcopy(vars(args))
        text_args.update(args.vision_language_emb["text"])
        text_args = Namespace(**text_args)
        self.text_embedding = Embedding(text_args)
        for embedding_name in text_args.embedding:
            tmp_emb = str2embedding[embedding_name](text_args, len(args.tokenizer.vocab))
            self.text_embedding.update(tmp_emb, embedding_name)

    def forward(self, src, seg=None):
        src_text, src_image, seg_text, seg_image, image_pos = src
        # image features
        with torch.no_grad():
            image_emb = self.vision_embedding(src_image, seg_image)
            image_emb = self.vision_encoder(image_emb, seg_image, output_layer=-2)[:,1:,:]
        image_emb = self.projection(image_emb)
        # text embedding
        text_emb = self.text_embedding(src_text, seg_text)
        # combine text and image
        if text_emb.shape[0] == 1:
            emb = torch.cat((text_emb[:,:image_pos[0],:], image_emb, text_emb[:,image_pos[0]:,:]), 1)
        else:
            emb = torch.cat((text_emb[0,:image_pos[0],:], image_emb[0], text_emb[0,image_pos[0]:,:]), 0).unsqueeze(0)
            for i in range(1, text_emb.shape[0]):
                tmp = torch.cat((text_emb[i,:image_pos[i],:], image_emb[i], text_emb[i,image_pos[i]:,:]), 0).unsqueeze(0)
                emb = torch.cat((emb, tmp), 0)

        return emb
