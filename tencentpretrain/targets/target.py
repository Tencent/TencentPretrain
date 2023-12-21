import torch
import torch.nn as nn

from tencentpretrain import mpu


class Target(nn.Module):
    def __init__(self):
        super(Target, self).__init__()
        self.target_name_list = []
        self.loss_info = {}

    def update(self, target, target_name):
        setattr(self, target_name, target)
        self.target_name_list.append(target_name)

    def forward(self, memory_bank, tgt, seg):
        self.loss_info = {}
        for i, target_name in enumerate(self.target_name_list):
            target = getattr(self, target_name)
            if len(self.target_name_list) > 1:
                self.loss_info[self.target_name_list[i]] = target(memory_bank, tgt[self.target_name_list[i]], seg)
            else:
                self.loss_info = target(memory_bank, tgt, seg)

        return self.loss_info


class TargetPipe(nn.Module):
    def __init__(self,args,model):
        super(TargetPipe, self).__init__()
        self.target_layer=model.target
    def forward(self,inputs):
        hidden, seg=inputs
        loss_info=self.target_layer(hidden, None, seg)
        
        return loss_info


def CrossEntropy(outputs, labels):
    output, loss_mask = outputs
    tgt_lm, seg = labels[0], labels[1]
    seg = seg.contiguous().view(-1)
    tgt_lm = tgt_lm.contiguous().view(-1)
    tgt_lm = tgt_lm[seg > loss_mask]
    losses = mpu.vocab_parallel_cross_entropy(output, tgt_lm)
    loss = torch.sum(losses.view(-1)) / len(losses)

    return loss
