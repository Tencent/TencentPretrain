import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from tencentpretrain.model_loader import _load_state_dict_into_model, load_model
from tencentpretrain.model_saver import save_model
from tencentpretrain.model_builder import build_model
from tencentpretrain.utils.logging import init_logger
from tencentpretrain.utils.optimizers import *
from tencentpretrain.utils import *
from tencentpretrain.utils.seed import set_seed


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    if args.data_processor == "mt":
        args.tgt_tokenizer = str2tokenizer[args.tgt_tokenizer](args, is_src=False)
        args.tgt_vocab = args.tgt_tokenizer.vocab

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.vocab = args.tokenizer.vocab

    # Build model.
    if args.deepspeed and args.enable_zero3:
        import deepspeed
        with deepspeed.zero.Init(config_dict_or_path=args.deepspeed_config):
            model_for_training = build_model(args)
    else:
        model_for_training = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        if args.deepspeed and args.enable_zero3:
            model_for_training = _load_state_dict_into_model(model_for_training, args.pretrained_model_path, "", args.lora_pretrained_model_path)
        else:
            model_for_training = load_model(model_for_training, args.pretrained_model_path,
                                        args.lora_pretrained_model_path)

    else:
        # Initialize with normal distribution.
        if args.deep_init:
            scaled_factor = 1 / math.sqrt(2.0 * args.layers_num)
            for n, p in list(model_for_training.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    if "linear_2.weight" in n or "final_linear.weight" in n:
                        p.data.normal_(0, 0.02 * scaled_factor)
                    elif "linear_2.bias" in n or "final_linear.bias" in n:
                        p.data.zero_()
                    else:
                        p.data.normal_(0, 0.02)
        else:
            for n, p in list(model_for_training.named_parameters()):
                if "gamma" not in n and "beta" not in n:
                    p.data.normal_(0, 0.02)

    if args.vqgan_model_path is not None:
        from tencentpretrain.utils.image_tokenizer import build_vqgan_model
        model_for_dataloader = build_vqgan_model(args)
    else:
        model_for_dataloader = None

    if args.deepspeed:
        worker(args.local_rank, None, args, model_for_training, model_for_dataloader)
    elif args.dist_train:
        # Multiprocessing distributed mode.
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model_for_training, model_for_dataloader), daemon=False)
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model_for_training, model_for_dataloader)
    else:
        # CPU mode.
        worker(None, None, args, model_for_training, model_for_dataloader)


class Trainer(object):
    def __init__(self, args):
        self.current_step = 1
        self.total_steps = args.total_steps
        self.accumulation_steps = args.accumulation_steps
        self.report_steps = args.report_steps
        self.save_checkpoint_steps = args.save_checkpoint_steps

        self.output_model_path = args.output_model_path

        self.start_time = time.time()
        self.total_loss = 0.0
        self.best_loss = float("inf")

        self.dist_train = args.dist_train
        self.batch_size = args.batch_size
        self.world_size = args.world_size
        self.logger = args.logger

    def forward_propagation(self, batch, model):

        raise NotImplementedError

    def report_and_reset_stats(self):

        raise NotImplementedError

    def train(self, args, gpu_id, rank, loader, model, optimizer, scheduler):
        model.train()
        loader_iter = iter(loader)
        while True:
            if self.current_step == self.total_steps + 1:
                break
            batch = list(next(loader_iter))
            self.seq_length = batch[0].size(1)
            if gpu_id is not None:
                for i in range(len(batch)):
                    if torch.is_tensor(batch[i]):
                        batch[i] = batch[i].cuda(gpu_id)

            loss = self.forward_propagation(batch, model)

            if args.deepspeed:
                model.backward(loss)
            else:
                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            if self.current_step % self.accumulation_steps == 0:
                if args.deepspeed:
                    model.step()
                else:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

            if self.current_step % self.report_steps == 0 and \
                    (not self.dist_train or (self.dist_train and rank == 0)):
                self.report_and_reset_stats()
                self.start_time = time.time()

            if args.deepspeed:
                if self.current_step % self.save_checkpoint_steps == 0:
                    if args.use_lora:
                        if rank == 0:
                            save_model(model, self.output_model_path + "-" + str(self.current_step), args.use_lora)
                    else:
                        model.save_checkpoint(self.output_model_path, str(self.current_step))
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        if args.use_lora:
                            if rank == 0:
                                save_model(model, self.output_model_path + "-" + str(self.current_step), args.use_lora)
                        else:
                            model.save_checkpoint(self.output_model_path, "-best")
            else:
                if self.current_step % self.save_checkpoint_steps == 0 and \
                        (not self.dist_train or (self.dist_train and rank == 0)):
                    save_model(model, self.output_model_path + "-" + str(self.current_step), args.use_lora)
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                        print("save best model! loss:" + str(self.best_loss))
                        save_model(model, self.output_model_path + "-best", args.use_lora)

            self.current_step += 1


class MlmTrainer(Trainer):
    def __init__(self, args):
        super(MlmTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class BertTrainer(Trainer):
    def __init__(self, args):
        super(BertTrainer, self).__init__(args)
        self.total_loss_sp = 0.0
        self.total_correct_sp = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_sp, seg = batch
        tgt = {"mlm": tgt_mlm, "sp": tgt_sp}
        loss_info = model(src, tgt, seg)
        loss_mlm, correct_mlm, denominator = loss_info["mlm"]
        loss_sp, correct_sp = loss_info["sp"]
        loss = loss_mlm + loss_sp
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_sp.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_sp.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_sp: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_sp: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_mlm / self.report_steps,
                  self.total_loss_sp / self.report_steps,
                  self.total_correct_mlm / self.total_denominator,
                  self.total_correct_sp / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


class AlbertTrainer(BertTrainer):
    pass


class LmTrainer(MlmTrainer):
    pass


class BilmTrainer(Trainer):
    def __init__(self, args):
        super(BilmTrainer, self).__init__(args)
        self.total_loss_forward, self.total_loss_backward = 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward = 0.0, 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_forward, tgt_backward, seg = batch
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        loss = loss_forward + loss_backward
        self.total_loss += loss.item()
        self.total_loss_forward += loss_forward.item()
        self.total_loss_backward += loss_backward.item()
        self.total_correct_forward += correct_forward.item()
        self.total_correct_backward += correct_backward.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_forward {:3.3f}"
              "| loss_backward {:3.3f}"
              "| acc_forward: {:3.3f}"
              "| acc_backward: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_forward / self.report_steps,
                  self.total_loss_backward / self.report_steps,
                  self.total_correct_forward / self.total_denominator,
                  self.total_correct_backward / self.total_denominator))

        self.total_loss, self.total_loss_forward, self.total_loss_backward = 0.0, 0.0, 0.0
        self.total_correct_forward, self.total_correct_backward, self.total_denominator = 0.0, 0.0, 0.0


class ClsTrainer(Trainer):
    def __init__(self, args):
        super(ClsTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_instances = 0.0

    def forward_propagation(self, batch, model):
        src, tgt, seg = batch
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps
        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_instances))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class MtTrainer(Trainer):
    def __init__(self, args):
        super(MtTrainer, self).__init__(args)
        self.total_correct = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_out, seg, tgt_in, tgt_seg = batch
        loss_info = model(src, tgt_out, seg, tgt_in, tgt_seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()

        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| acc: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_correct / self.total_denominator))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_denominator = 0.0


class ClsMlmTrainer(Trainer):
    def __init__(self, args):
        super(ClsMlmTrainer, self).__init__(args)
        self.total_loss_cls = 0.0
        self.total_correct_cls = 0.0
        self.total_instances = 0.0

        self.total_loss_mlm = 0.0
        self.total_correct_mlm = 0.0
        self.total_denominator = 0.0

    def forward_propagation(self, batch, model):
        src, tgt_mlm, tgt_cls, seg = batch
        tgt = {"mlm": tgt_mlm, "cls": tgt_cls}
        loss_info = model(src, tgt, seg)
        loss_mlm, correct_mlm, denominator = loss_info["mlm"]
        loss_cls, correct_cls = loss_info["cls"]
        loss = loss_mlm + loss_cls
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_cls += loss_cls.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_cls += correct_cls.item()
        self.total_denominator += denominator.item()
        self.total_instances += src.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        self.logger.info("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_cls: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_cls: {:3.3f}".format(
                  self.current_step,
                  self.total_steps,
                  done_tokens / (time.time() - self.start_time),
                  self.total_loss / self.report_steps,
                  self.total_loss_mlm / self.report_steps,
                  self.total_loss_cls / self.report_steps,
                  self.total_correct_mlm / self.total_denominator,
                  self.total_correct_cls / self.total_instances))

        self.total_loss, self.total_loss_mlm, self.total_loss_cls = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_cls, self.total_instances = 0.0, 0.0


class T5Trainer(MtTrainer):
    pass


class GsgTrainer(MtTrainer):
    pass


class BartTrainer(MtTrainer):
    pass


class PrefixlmTrainer(MlmTrainer):
    pass


class VitTrainer(ClsTrainer):
    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size
        self.logger.info("| {:8d}/{:8d} steps"
                         "| {:8.2f} patches/s"
                         "| loss {:7.2f}"
                         "| acc: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_correct / self.total_instances))

        self.total_loss = 0.0
        self.total_correct = 0.0
        self.total_instances = 0.0


class ViltTrainer(BertTrainer):
    def forward_propagation(self, batch, model):
        src_text, src_image, tgt_mlm, tgt_match, seg = batch
        tgt = {"mlm": tgt_mlm, "sp": tgt_match}
        loss_info = model((src_text, src_image), tgt, seg)
        loss_mlm, correct_mlm, denominator = loss_info["mlm"]
        loss_match, correct_match = loss_info["sp"]
        loss = loss_mlm + loss_match
        self.total_loss += loss.item()
        self.total_loss_mlm += loss_mlm.item()
        self.total_loss_sp += loss_match.item()
        self.total_correct_mlm += correct_mlm.item()
        self.total_correct_sp += correct_match.item()
        self.total_denominator += denominator.item()
        self.total_instances += src_text.size(0)
        loss = loss / self.accumulation_steps

        return loss

    def report_and_reset_stats(self):
        done_tokens = self.batch_size * self.seq_length * self.report_steps
        if self.dist_train:
            done_tokens *= self.world_size

        print("| {:8d}/{:8d} steps"
              "| {:8.2f} tokens/s"
              "| loss {:7.2f}"
              "| loss_mlm: {:3.3f}"
              "| loss_match: {:3.3f}"
              "| acc_mlm: {:3.3f}"
              "| acc_match: {:3.3f}".format(
            self.current_step,
            self.total_steps,
            done_tokens / (time.time() - self.start_time),
            self.total_loss / self.report_steps,
            self.total_loss_mlm / self.report_steps,
            self.total_loss_sp / self.report_steps,
            self.total_correct_mlm / self.total_denominator,
            self.total_correct_sp / self.total_denominator))

        self.total_loss, self.total_loss_mlm, self.total_loss_sp = 0.0, 0.0, 0.0
        self.total_correct_mlm, self.total_denominator = 0.0, 0.0
        self.total_correct_sp, self.total_instances = 0.0, 0.0


class ClipTrainer(ClsTrainer):
    def forward_propagation(self, batch, model):
        src_text, src_img, seg_text, seg_img = batch
        loss_info = model((src_text, src_img), None, (seg_text, seg_img))
        loss, correct = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_instances += src_text.size(0)
        loss = loss / self.accumulation_steps
        return loss


class S2tTrainer(MtTrainer):
    pass


class BeitTrainer(MlmTrainer):
    def forward_propagation(self, batch,  model):
        src, tgt, seg, mask = batch
        loss_info = model((src, mask), tgt, seg)
        loss, correct, denominator = loss_info
        self.total_loss += loss.item()
        self.total_correct += correct.item()
        self.total_denominator += denominator.item()
        loss = loss / self.accumulation_steps
        return loss


class DalleTrainer(LmTrainer):
    pass


class AlpacaTrainer(LmTrainer):
    pass


str2trainer = {"bert": BertTrainer, "mlm": MlmTrainer, "lm": LmTrainer,
               "albert": AlbertTrainer, "bilm": BilmTrainer, "cls": ClsTrainer,
               "mt": MtTrainer, "t5": T5Trainer, "gsg": GsgTrainer,
               "bart": BartTrainer, "prefixlm": PrefixlmTrainer, "cls_mlm": ClsMlmTrainer,
               "vit": VitTrainer, "vilt": ViltTrainer, "clip": ClipTrainer, "s2t": S2tTrainer,
               "beit": BeitTrainer, "dalle": DalleTrainer, "alpaca": AlpacaTrainer}


def worker(proc_id, gpu_ranks, args, model_for_training, model_for_dataloader=None):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    set_seed(args.seed)

    # Get logger
    args.logger = init_logger(args)

    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed(dist_backend=args.backend)
        rank = dist.get_rank()
        gpu_id = proc_id
    elif args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    # Build optimizer.
    param_optimizer = list(model_for_training.named_parameters())
    if args.use_lora:
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if 'lora' in n]}
        ]
        for n, p in list(model_for_training.named_parameters()):
            if 'lora' not in n:
                p.requires_grad = False
    else:
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

    if args.optimizer in ["adamw"]:
        if args.deepspeed and deepspeed.__version__ > "0.5.8":
            custom_optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False)
        else:
            custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        custom_optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup)
    elif args.scheduler in ["tri_stage"]:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup, args.total_steps*args.decay, args.total_steps)
    else:
        custom_scheduler = str2scheduler[args.scheduler](custom_optimizer, args.total_steps*args.warmup, args.total_steps)

    if args.deepspeed:
        model_for_training, optimizer, _, scheduler = deepspeed.initialize(
                                                    model=model_for_training,
                                                    model_parameters=optimizer_grouped_parameters,
                                                    args=args,
                                                    optimizer=custom_optimizer,
                                                    lr_scheduler=custom_scheduler,
                                                    mpu=None,
                                                    dist_init_required=False)
    else:
        if gpu_id is not None:
            model_for_training.cuda(gpu_id)
            if model_for_dataloader is not None:
                model_for_dataloader.cuda(gpu_id)
        optimizer = custom_optimizer
        scheduler = custom_scheduler
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model_for_training, optimizer = amp.initialize(model_for_training, optimizer, opt_level=args.fp16_opt_level)
            args.amp = amp

        if args.dist_train:
            # Initialize multiprocessing distributed training environment.
            dist.init_process_group(backend=args.backend,
                                    init_method=args.master_ip,
                                    world_size=args.world_size,
                                    rank=rank)
            model_for_training = DistributedDataParallel(model_for_training, device_ids=[gpu_id], find_unused_parameters=True)
            if model_for_dataloader is not None:
                model_for_dataloader = DistributedDataParallel(model_for_dataloader, device_ids=[gpu_id], find_unused_parameters=False)
            args.logger.info("Worker %d is training ... " % rank)
        else:
            args.logger.info("Worker is training ...")

    if args.dist_train:
        if model_for_dataloader is not None:
            model_for_dataloader = model_for_dataloader.module
        train_loader = str2dataloader[args.data_processor](args, args.dataset_path, args.batch_size, rank, args.world_size, gpu_id, True, model_for_dataloader)
    else:
        train_loader = str2dataloader[args.data_processor](args, args.dataset_path, args.batch_size, 0, 1, gpu_id, True, model_for_dataloader)


    trainer = str2trainer[args.data_processor](args)
    trainer.train(args, gpu_id, rank, train_loader, model_for_training, optimizer, scheduler)
