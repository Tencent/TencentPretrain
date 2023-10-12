
## Low-Rank Adaptation (LoRA)

Use `--use_lora` to enable lora training. At this time, the model will only save part of the weight of lora.
```
python3 pretrain.py --pretrained_model_path models/gpt2.bin \ 
                    --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --config_path models/gpt2/config.json \
                    --output_model_path models/output_model_loar.bin \
                    --world_size 8  --learning_rate 1e-4 \
                    --data_processor lm --use_lora --lora_dropout 0.05
```

Load pre-trained model and pre-trained lora model for post-training.

```
python3 pretrain.py -pretrained_model_path models/gpt2.bin \ 
                    --lora_pretrained_model_path models/output_model_loar.bin \
                    --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                    --config_path models/gpt2/config.json \
                    --output_model_path models/output_model_loar.bin \
                    --world_size 8  --learning_rate 1e-4 \
                    --data_processor lm --use_lora --lora_dropout 0.05
```

## [Deepspeed Zero-3 Offload](https://www.deepspeed.ai/2021/03/07/zero3-offload.html)

Use `--enable_zero3` to enable Zero-3 Offload.

```
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_zero3_config.json \
                      --pretrained_model_path models/gpt2.bin \ 
                      --dataset_path dataset.pt --vocab_path models/google_zh_vocab.txt \
                      --config_path models/gpt2/config.json \
                      --world_size 8  --learning_rate 1e-4 \
                      --data_processor lm --enable_zero3
```