import os
import random
import pickle
import torch
from tencentpretrain.utils.constants import *
from tencentpretrain.utils.tokenizers import *
from tencentpretrain.utils.mask import mask_seq
from tencentpretrain.utils.augment import SpecAugment


class Dataloader(object):
    def __init__(self, args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle=False, model_for_dataloader=None):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.rank = rank
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.shuffle = shuffle
        self.model_for_dataloader = model_for_dataloader
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.world_size == self.rank:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()


class BertDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))

                if len(ins) == 4:
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [0] * pad_num)
                else:
                    src_single, tgt_mlm_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(src_single))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [0] * pad_num)

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)


class MlmDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))

                if len(ins) == 3:
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[2][0] + [0] * pad_num)
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_single)
                    src.append(src_single)
                    tgt.append([0] * len(src_single))
                    for mask in tgt_single:
                        tgt[-1][mask[0]] = mask[1]
                    seg.append([1] * ins[1][0] + [0] * pad_num)

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class AlbertDataloader(BertDataloader):
    '''
    AlbertDataloader can reuse the code of BertDataloader.
    '''
    pass


class LmDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                src.append(src_single[:-1])
                tgt.append(src_single[1:])
                seg.append([1] * ins[1][0] + [0] * (len(src_single) - 1 - ins[1][0]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class BilmDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_forward = []
            tgt_backward = []
            seg = []

            for ins in instances:
                src_single, pad_num = ins[0]
                tgt_forward_single, tgt_backward_single = ins[1], ins[2]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                    tgt_forward_single.append(self.vocab.get(PAD_TOKEN))
                    tgt_backward_single.append(self.vocab.get(PAD_TOKEN))
                src.append(src_single)
                tgt_forward.append(tgt_forward_single)
                tgt_backward.append(tgt_backward_single)
                seg.append([1] * ins[3][0] + [0] * (len(src_single) - ins[3][0]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_forward), \
                torch.LongTensor(tgt_backward), \
                torch.LongTensor(seg)


class MtDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []
            tgt_seg = []

            for ins in instances:
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                tgt_single, pad_num = ins[1]
                for _ in range(pad_num):
                    tgt_single.append(self.vocab.get(PAD_TOKEN))

                src.append(src_single)
                tgt_in.append(tgt_single[:-1])
                tgt_out.append(tgt_single[1:])
                seg.append([1] * ins[2][0] + [0] * (len(src_single) - ins[2][0]))
                pad_num = max(ins[1][1] - 1, 0)  # left shifted, pad_num >= 0
                tgt_seg.append([1] * (len(tgt_in[-1]) - pad_num) + [0] * pad_num)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_seg)


class T5Dataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []
            tgt_seg = []

            tgt_seq_length = 0

            for _, ins in enumerate(instances):
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))

                if len(ins) == 3:
                    tgt_single = ins[1]
                    seg.append([1] * ins[2][0] + [0] * pad_num)
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    seg.append([1] * ins[1][0] + [0] * pad_num)

                MASK_ID = self.vocab.get(MASK_TOKEN)
                SENTINEL_ID = self.vocab.get(SENTINEL_TOKEN)
                PAD_ID = self.vocab.get(PAD_TOKEN)

                for src_index, _ in tgt_single:
                    if src_single[src_index] != MASK_ID:
                        src_single[src_index] = MASK_ID

                tgt_in_single = [self.vocab.get(CLS_TOKEN)]
                mask_index = 0
                src_with_sentinel = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_sentinel) > 0 and src_with_sentinel[-1] == (SENTINEL_ID - 1):
                            pass
                        else:
                            src_with_sentinel.append(SENTINEL_ID)
                            tgt_in_single.append(SENTINEL_ID)
                            if SENTINEL_ID < len(self.vocab) - 1:
                                SENTINEL_ID += 1
                        tgt_in_single.append(tgt_single[mask_index][1])
                        mask_index += 1
                    else:
                        src_with_sentinel.append(token_id)
                tgt_in_single.append(SENTINEL_ID)
                tgt_in_single.append(self.vocab.get(SEP_TOKEN))

                tgt_seg_single = [1] * len(tgt_in_single)

                while len(src_with_sentinel) < len(src_single):
                    src_with_sentinel.append(PAD_ID)

                if len(tgt_in_single) > tgt_seq_length:
                    tgt_seq_length = len(tgt_in_single)

                src.append(src_with_sentinel)
                tgt_in.append(tgt_in_single)
                tgt_seg.append(tgt_seg_single)
                tgt_out.append(tgt_in[-1][1:] + [PAD_ID])

            for i in range(len(tgt_in)):
                while len(tgt_in[i]) != tgt_seq_length:
                    tgt_in[i].append(PAD_ID)
                    tgt_out[i].append(PAD_ID)
                    tgt_seg[i].append(0)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_seg)


class GsgDataloader(MtDataloader):
    pass


class BartDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_in = []
            tgt_out = []
            seg = []
            tgt_seg = []

            for _, ins in enumerate(instances):
                src_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                tgt_single, pad_num = ins[1]
                for _ in range(pad_num):
                    tgt_single.append(self.vocab.get(PAD_TOKEN))

                src_single, _ = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking,
                                         self.span_geo_prob, self.span_max_length)
                seg_pos = ins[2][0]
                tgt_in.append(tgt_single[:-1])
                tgt_out.append(tgt_single[1:])
                pad_num = max(ins[1][1] - 1, 0)  # left shifted, pad_num >= 0
                tgt_seg.append([1] * (len(tgt_in[-1]) - pad_num) + [0] * pad_num)


                MASK_ID = self.vocab.get(MASK_TOKEN)

                src_with_span_mask = []
                for token_id in src_single:
                    if token_id == MASK_ID:
                        if len(src_with_span_mask) > 0 and src_with_span_mask[-1] == MASK_ID:
                            seg_pos -= 1
                        else:
                            src_with_span_mask.append(MASK_ID)
                    else:
                        src_with_span_mask.append(token_id)

                while len(src_with_span_mask) < len(src_single):
                    src_with_span_mask.append(self.vocab.get(PAD_TOKEN))

                seg.append([1] * seg_pos + [0] * (len(src_single) - seg_pos))
                src.append(src_with_span_mask)


            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_out), \
                torch.LongTensor(seg), \
                torch.LongTensor(tgt_in), \
                torch.LongTensor(tgt_seg)


class ClsDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src_single, pad_num = ins[0]
                seg_pos_single = ins[2]

                if len(seg_pos_single) == 1:
                    seg_single = [1] * seg_pos_single[0]
                elif len(seg_pos_single) == 2:
                    seg_single = [1] * seg_pos_single[0] + [2] * seg_pos_single[1]
                
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                    seg_single.append(0)
                
                src.append(src_single)
                tgt.append(ins[1])
                seg.append(seg_single)

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class PrefixlmDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:
                src_single, pad_num = ins[0]
                tgt_single = ins[1]
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                    tgt_single.append(self.vocab.get(PAD_TOKEN))
                src.append(src_single)
                tgt.append(tgt_single)
                seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [0] * (len(src_single) - ins[2][1]))

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt), \
                torch.LongTensor(seg)


class ClsMlmDataloader(Dataloader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            tgt_cls = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                src_single, pad_num = ins[0]
                seg_pos_single = ins[-1]
                tgt_cls.append(ins[-2])

                if len(seg_pos_single) == 1:
                    seg_single = [1] * seg_pos_single[0]
                elif len(seg_pos_single) == 2:
                    seg_single = [1] * seg_pos_single[0] + [2] * seg_pos_single[1]
                
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                    seg_single.append(0)
                seg.append(seg_single)

                if len(ins) == 4 :
                    src.append(src_single)
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(src_single))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                else:
                    src_single, tgt_single = mask_seq(src_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    src.append(src_single)
                    masked_words_num += len(tgt_single)
                    tgt_mlm.append([0] * len(src_single))
                    for mask in tgt_single:
                        tgt_mlm[-1][mask[0]] = mask[1]

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(tgt_cls), \
                torch.LongTensor(seg)


class VisionDataloader(Dataloader):
    def __init__(self, args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle=False, model_for_dataloader=None):
        super(VisionDataloader, self).__init__(args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle, model_for_dataloader)
        self.patch_size = args.patch_size
        self.image_height = args.image_height
        self.image_width = args.image_width

        from torchvision import transforms
        from tencentpretrain.utils.misc import ZeroOneNormalize

        preprocess_pipeline = []
        if "corp" in args.image_preprocess:
            preprocess_pipeline.append(transforms.RandomResizedCrop(max(self.image_height, self.image_width)))
        if "horizontal_flip" in args.image_preprocess:
            preprocess_pipeline.append(transforms.RandomHorizontalFlip())
        preprocess_pipeline.append(transforms.Resize((self.image_height, self.image_width)))
        preprocess_pipeline.append(ZeroOneNormalize())
        if "normalize" in args.image_preprocess:
            preprocess_pipeline.append(transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.transform = transforms.Compose(preprocess_pipeline)


class VitDataloader(VisionDataloader):
    def __iter__(self):
        """
        instances: (tgt, image_path)
            tgt: The category the image belongs to
            image_path: Path of the image sample

        Returns:
            src_image: [batch_size x channel_size x width x hight]
            seg: [batch_size x (patch_num + 1)]
            tgt: [batch_size]
        """
        from torchvision.io import read_image
        from torchvision.io.image import ImageReadMode
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []

            for ins in instances:

                image = read_image(ins[1], ImageReadMode.RGB)
                image = image.cuda(self.gpu_id)
                src.append(self.transform(image))
                tgt.append(ins[0])
                seg.append([1] * ((self.image_height // self.patch_size) * (self.image_width // self.patch_size) + 1))

            yield torch.stack(src, 0), \
                  torch.LongTensor(tgt), \
                  torch.LongTensor(seg)


class ViltDataloader(VisionDataloader):
    def __iter__(self):
        """
        instances: (src_text, seg_text, image_path)
            src_text: Tokens of the text sample
            seg_text: Segment input of text sample
            src_image: Path of the image sample

        Returns:
            src_text: [batch_size x seq_length]
            src_image: [batch_size x channel_size x width x hight]
            tgt_mlm: [batch_size x (seq_length + patch_num + 1)]
            tgt_match: [batch_size]
            seg: [batch_size x (seq_length + patch_num + 1)]
        """
        from torchvision.io import read_image
        from torchvision.io.image import ImageReadMode
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src_text = []
            src_image = []
            tgt_mlm = []
            tgt_match = []
            seg = []

            masked_words_num = 0

            for ins in instances:
                src_text_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_text_single.append(self.vocab.get(PAD_TOKEN))
                src_text_single, tgt_mlm_single = mask_seq(src_text_single, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                src_text.append(src_text_single)
                masked_words_num += len(tgt_mlm_single)
                tgt_mlm.append([0] * len(src_text_single))
                for mask in tgt_mlm_single:
                    tgt_mlm[-1][mask[0]] = mask[1]

                if random.random() < 0.5:
                    image = read_image(ins[2], ImageReadMode.RGB)
                    tgt_match.append(1)
                else:
                    image = read_image(random.choice(self.buffer)[2], ImageReadMode.RGB)
                    tgt_match.append(0)

                seg_image = [2] * ((self.image_height // self.patch_size) * (self.image_width // self.patch_size) + 1)
                tgt_mlm[-1].extend([0] * len(seg_image))
                image = image.cuda(self.gpu_id)
                src_image_single = self.transform(image)
                src_image.append(src_image_single)
                seg.append([1] * ins[1][0] + [0] * pad_num + seg_image)

            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src_text), \
                  torch.stack(src_image, 0), \
                  torch.LongTensor(tgt_mlm), \
                  torch.LongTensor(tgt_match), \
                  torch.LongTensor(seg)


class ClipDataloader(VisionDataloader):

    def __iter__(self):
        """
        instances: (src_text, src_image, seg_text)
            src_text: Tokens of the text sample
            src_image: Path of the image sample
            seg_text: Segment input of text sample

        Returns:
            src_text: [batch_size x seq_length]
            src_image: [batch_size x channel_size x width x hight]
            seg_text: [batch_size x seq_length]
            seg_image: [batch_size x (patch_num + 1)]
        """
        from torchvision.io import read_image
        from torchvision.io.image import ImageReadMode
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src_text = []
            src_image = []
            seg_text = []
            seg_image = []
            for ins in instances:
                src_text_single, pad_num = ins[0]
                for _ in range(pad_num):
                    src_text_single.append(self.vocab.get(PAD_TOKEN))

                src_text.append(src_text_single)
                seg_text.append([1] * ins[1][0] + [0] * pad_num)
                image = read_image(ins[2], ImageReadMode.RGB)
                image = image.cuda(self.gpu_id)
                src_image.append(self.transform(image))
                seg_image.append([1] * ((self.image_height // self.patch_size) * (self.image_width // self.patch_size) + 1))

            yield  torch.LongTensor(src_text), \
                   torch.stack(src_image, 0), \
                   torch.LongTensor(seg_text), \
                   torch.LongTensor(seg_image)


class AudioDataloader(Dataloader):
    def __init__(self, args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle=False, model_for_dataloader=None):
        super(AudioDataloader, self).__init__(args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle, model_for_dataloader)
        self.dataset_folder = os.path.dirname(dataset_path)
        self.sampling_rate = args.sampling_rate
        self.normalize_means, self.normalize_vars, self.ceptral_normalize = True, True, True
        self.padding_value = 0.0
        self.audio_feature_size = args.audio_feature_size
        self.conv_layers_num = args.conv_layers_num
        self.max_audio_frames = args.max_audio_frames
        self.specaugment = None

        if "normalize_means" not in args.audio_preprocess:
            self.normalize_means = False
        if "normalize_vars" not in args.audio_preprocess:
            self.normalize_vars = False
        if "ceptral_normalize" not in args.audio_preprocess:
            self.ceptral_normalize = False
        if "sepcaugment" in args:
            self.specaugment = SpecAugment(args)

def utterance_cmvn(x, normalize_means=True, normalize_vars=True, gpu_id=None):
    mean = x.mean(axis=0)
    square_sums = (x ** 2).sum(axis=0)

    if normalize_means:
        x = torch.sub(x, mean)
    if normalize_vars:
        var = square_sums / x.size(0) - mean ** 2
        if gpu_id is not None:
            std = torch.sqrt(torch.maximum(var, torch.full(var.size(), 1e-10).cuda(gpu_id)))
        else:
            std = torch.sqrt(torch.maximum(var, torch.full(var.size(), 1e-10)))
        x = torch.div(x, std)

    return x


class S2tDataloader(AudioDataloader):

    def __iter__(self):
        import torchaudio
        import torchaudio.compliance.kaldi as ta_kaldi

        padding_vector = torch.FloatTensor(self.audio_feature_size * [self.padding_value] if self.audio_feature_size > 1 else self.padding_value).unsqueeze(0).cuda(self.gpu_id)
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            tgt_in = []
            tgt_out = []
            src_audio = []
            seg_audio = []
            tgt_seg = []

            for ins in instances:
                text_single, pad_num = ins[0]
                for _ in range(pad_num):
                    text_single.append(self.vocab.get(PAD_TOKEN))

                waveform, _ = torchaudio.load(ins[2])  # waveform, sample_rate
                waveform = waveform * (2 ** 15)  # Kaldi compliance: 16-bit signed integers
                waveform = waveform.cuda(self.gpu_id)
                feature = ta_kaldi.fbank(waveform, num_mel_bins=self.audio_feature_size,
                                         sample_frequency=self.sampling_rate)
                if self.ceptral_normalize:
                    feature = utterance_cmvn(feature, self.normalize_means, self.normalize_vars, self.gpu_id)
                difference = self.max_audio_frames - feature.size(0)
                if difference < 0:
                    continue
                else:
                    src_audio.append(torch.cat([feature] + [padding_vector] * difference))

                src_pad_num = int(self.max_audio_frames / self.conv_layers_num / 2) - int(feature.size(0) / self.conv_layers_num / 2)
                seg_audio.append([1] * int(feature.size(0) / self.conv_layers_num / 2) + [0] * src_pad_num)
                tgt_out.append(text_single[1:])
                text_single[-pad_num-1] = self.vocab.get(PAD_TOKEN)

                tgt_in.append(text_single[:-1])
                pad_num = max(pad_num - 1, 0)  # left shifted, pad_num >= 0
                tgt_seg.append([1] * (len(tgt_in[-1]) - pad_num) + [0] * pad_num)

            if len(src_audio) == 0:
                continue
            if self.specaugment:
                src_audio = self.specaugment(src_audio)

            yield  torch.stack(src_audio, 0), \
                   torch.LongTensor(tgt_out), \
                   torch.LongTensor(seg_audio), \
                   torch.LongTensor(tgt_in), \
                   torch.LongTensor(tgt_seg)


class BeitDataloader(VisionDataloader):

    def __init__(self, args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle=False, model_for_dataloader=None):
        super(BeitDataloader, self).__init__(args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle, model_for_dataloader)
        from tencentpretrain.utils.image_tokenizer import build_vqgan_model
        self.vqgan = self.model_for_dataloader


    def mask(self, image_tokens, mask_rate = 0.15):
        mask_num = int(len(image_tokens) * mask_rate)
        mask_index = random.sample(range(1, len(image_tokens)), mask_num)
        tgt = [0] * len(image_tokens)
        for idx in mask_index:
            tgt[idx] = image_tokens[idx]
        return tgt, mask_index


    def __iter__(self):
        """
        instances: (tgt, image_path)
            tgt: The category the image belongs to
            image_path: Path of the image sample

        Returns:
            src_image: [batch_size x channel_size x width x hight]
            seg: [batch_size x (patch_num + 1)]
            tgt: [batch_size]
        """
        from torchvision.io import read_image
        from torchvision.io.image import ImageReadMode
        from tencentpretrain.utils.image_tokenizer import image_tokenize

        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            mask = []
            for ins in instances:

                image = read_image(ins, ImageReadMode.RGB)
                image = image.cuda(self.gpu_id)
                image = self.transform(image)
                src.append(image)
                image_tokens = [0] + image_tokenize(self.vqgan, image)
                tgt_single, mask_index = self.mask(image_tokens)
                tgt.append(tgt_single)
                mask.append(mask_index)
                seg.append([1] * ((self.image_height // self.patch_size) * (self.image_width // self.patch_size) + 1))

            yield torch.stack(src, 0), \
                  torch.LongTensor(tgt), \
                  torch.LongTensor(seg), \
                  mask


class DalleDataloader(VisionDataloader):

    def __init__(self, args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle=False, model_for_dataloader=None):
        super(DalleDataloader, self).__init__(args, dataset_path, batch_size, rank, world_size, gpu_id, shuffle, model_for_dataloader)
        from tencentpretrain.utils.image_tokenizer import build_vqgan_model
        self.vqgan = self.model_for_dataloader
        self.vocab_bias = args.tokenizer.vocab_bias


    def __iter__(self):
        from torchvision.io import read_image
        from torchvision.io.image import ImageReadMode
        from tencentpretrain.utils.image_tokenizer import image_tokenize

        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt = []
            seg = []
            for ins in instances:
                src_single, pad_num = ins[0]

                image = read_image(ins[2], ImageReadMode.RGB)
                image = image.cuda(self.gpu_id)
                image = self.transform(image)
                image_tokens = [i + self.vocab_bias for i in image_tokenize(self.vqgan, image)]
                src_single.extend(image_tokens)
                for _ in range(pad_num):
                    src_single.append(self.vocab.get(PAD_TOKEN))
                seg_single = [1] * ins[1][0] + [2] * len(image_tokens) + [0] * pad_num
                src.append(src_single)
                tgt.append(src_single[1:] + [self.vocab.get(SEP_TOKEN)])
                seg.append(seg_single)

            yield torch.LongTensor(src), \
                  torch.LongTensor(tgt), \
                  torch.LongTensor(seg)


class AlpacaDataloader(LmDataloader):
    pass
