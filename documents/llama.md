
## Training

1. Clone the TencentPretrain project and install dependencies: PyTorch, DeepSpeed, SentencePiece

```
git clone https://github.com/Tencent/TencentPretrain.git
```

2. Convert LLaMA-7B weights to TencentPretrain format

```
cd TencentPretrain

python3 scripts/convert_llama_to_tencentpretrain.py --input_model_path $LLaMA_7B_FOLDER/consolidated.00.pth --output_model_path models/llama-7b.bin --layers_num 32
```

3. Modify configuration file

Check out the `tencentpretrain/utils/constants.py` file, and modify L4: `special_tokens_map.json` to `llama_special_tokens_map.json`

4. Data preprocess. We use the example corpus in the project for pre-training, one can also use custom data training in the same format (one sample per line).

```
python3 preprocess.py --corpus_path corpora/book_review.txt --spm_model_path $LLaMA_7B_FOLDER/tokenizer.model \
                      --dataset_path dataset.pt --processes_num 8 --data_processor lm
```

5. Start training.

```
deepspeed pretrain.py --deepspeed --deepspeed_config models/deepspeed_config.json \
                      --pretrained_model_path models/llama-7b.bin \
                      --dataset_path dataset.pt --spm_model_path $LLaMA_7B_FOLDER/tokenizer.model \
                      --config_path models/llama/7b_config.json \
                      --output_model_path models/output_model.bin \
                      --world_size 8 --learning_rate 1e-4  \
                      --data_processor lm --total_steps 10000 --save_checkpoint_steps 2000 --batch_size 24
```

## Inference

Similar to facebookresearch/llama, TencentPretrain also provides language model inference code. 
For example, using a single GPU for LLaMA-7B inference, the prompt is in the file `beginning.txt`:

```
python3 scripts/generate_lm.py --load_model_path models/llama-7b.bin --spm_model_path $LLaMA_7B_FOLDER/tokenizer.model \
                               --test_path beginning.txt --prediction_path generated_sentence.txt \
                               --config_path models/llama/7b_config.json 
```

For now, TencentPretrain only support LLaMA-7B training. We are working on our framework to support LLaMA model training/fine-tuning at all scales and sharing more experimental results.