import torch
import torch.nn as nn
import torch.nn.functional as F


class NerTarget(nn.Module):
    def __init__(self, args, vocab_size):
        super(NerTarget, self).__init__()
        self.labels_num = 3
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.crf_target = False
        if self.crf_target:
            from torchcrf import CRF
            self.crf = CRF(self.labels_num, batch_first=True)
            self.seq_length = args.seq_length

    def forward(self, memory_bank, tgt, seg):
        logits = self.output_layer(memory_bank)
        if self.crf_target:
            tgt_mask = seg.type(torch.uint8)
            pred = self.crf.decode(logits, mask=tgt_mask)
            for j in range(len(pred)):
                while len(pred[j]) < self.seq_length:
                    pred[j].append(self.labels_num - 1)
            pred = torch.tensor(pred).contiguous().view(-1)

            loss = -self.crf(F.log_softmax(logits, 2), tgt, mask=tgt_mask, reduction='mean')


        else:
            tgt_mask = seg.contiguous().view(-1).float()
            logits = logits.contiguous().view(-1, self.labels_num)
            pred = logits.argmax(dim=-1)

            tgt = tgt.contiguous().view(-1, 1)
            one_hot = torch.zeros(tgt.size(0), self.labels_num). \
                to(torch.device(tgt.device)). \
                scatter_(1, tgt, 1.0)
            numerator = -torch.sum(nn.LogSoftmax(dim=-1)(logits) * one_hot, 1)
            numerator = torch.sum(tgt_mask * numerator)
            denominator = torch.sum(tgt_mask) + 1e-6
            loss = numerator / denominator

        correct, gold_entities_num, pred_entities_num = 0, 0, 0
        pred_entities_pos, gold_entities_pos = set(), set()

        for j in range(tgt.size()[0]):
            if tgt[j].item() == 1:
                gold_entities_num += 1

        for j in range(pred.size()[0]):
            if pred[j].item() == 1:
                pred_entities_num += 1

        for j in range(tgt.size()[0]):
            if tgt[j].item() == 1:
                start = j
                for k in range(j + 1, tgt.size()[0]):
                    if tgt[k].item() == 0 or tgt[k].item() == 1:
                        end = k - 1
                        break
                else:
                    end = tgt.size()[0] - 1
                gold_entities_pos.add((start, end))

        for j in range(pred.size()[0]):
            if pred[j].item() == 1 and tgt[j].item() != 0:
                start = j
                for k in range(j + 1, pred.size()[0]):
                    if pred[k].item() == 0 or pred[k].item() == 1:
                        end = k - 1
                        break
                else:
                    end = pred.size()[0] - 1
                pred_entities_pos.add((start, end))

        for entity in pred_entities_pos:
            if entity not in gold_entities_pos:
                continue
            for j in range(entity[0], entity[1] + 1):
                if tgt[j].item() != pred[j].item():
                    break
            else:
                correct += 1

        return loss, correct, denominator
