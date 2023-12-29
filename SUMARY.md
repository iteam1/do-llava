# Concepts

- [LLaVA-Plus](https://llava-vl.github.io/llava-plus/): Learning to Use Tools for Creating Multimodal Agents, with LLaVA-Plus (LLaVA that Plug and Learn to Use Skills)

- `LLaVA-1.5`: with LoRA achieves comparable performance as full-model finetuning, with a reduced GPU RAM requirement ([ckpts](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md#llava-v15), [script](https://github.com/haotian-liu/LLaVA#train)). We also provide a [doc](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md) on how to finetune LLaVA-1.5 on your own dataset with LoRA.

# Install

- Clone this repository and navigate to LLaVA folder

            git clone https://github.com/haotian-liu/LLaVA.git
            cd LLaVA

- Install Package

            conda create -n llava python=3.10 -y
            conda activate llava
            pip install --upgrade pip  # enable PEP 660 support
            pip install -e .

- Install additional packages for training cases

            pip install -e ".[train]"
            pip install flash-attn --no-build-isolation

# Deploy

## Gradio Web UI

- Launch a controller:

            python -m llava.serve.controller \
            --host 0.0.0.0 \
            --port 10000`   

- Launch a gradio web server:

            python -m llava.serve.gradio_web_server \
            --controller http://localhost:10000 \
            --model-list-mode reload`

- Launch a model worker:

            python -m llava.serve.model_worker \
            --host 0.0.0.0 \
            --controller http://localhost:10000 \
            --port 40000 \
            --worker http://localhost:40000 \
            --model-path liuhaotian/llava-v1.5-13b

- Lauch other model worker:

            python -m llava.serve.model_worker \
            --host 0.0.0.0 \
            --controller http://localhost:10000 \
            --port <different from 40000, say 40001> \
            --worker http://localhost:<change accordingly, i.e. 40001> \
            --model-path <ckpt2>

- Launch a model worker (Multiple GPUs, when GPU VRAM <= 24GB):

            CUDA_VISIBLE_DEVICES=0,1 python -m llava.serve.model_worker \
            --host 0.0.0.0 \
            --controller http://localhost:10000 \
            --port 40000 \
            --worker http://localhost:40000 \
            --model-path liuhaotian/llava-v1.5-13b

- Launch a model worker (4-bit, 8-bit inference, quantized):

            python -m llava.serve.model_worker \
            --host 0.0.0.0 \
            --controller http://localhost:10000 \
            --port 40000 \
            --worker http://localhost:40000 \
            --model-path liuhaotian/llava-v1.5-13b \
            --load-4bit

- Launch a model worker (LoRA weights, unmerged):

            python -m llava.serve.model_worker \
            --host 0.0.0.0 \
            --controller http://localhost:10000 \
            --port 40000 \
            --worker http://localhost:40000 \
            --model-path liuhaotian/llava-v1-0719-336px-lora-vicuna-13b-v1.3 \
            --model-base lmsys/vicuna-13b-v1.3

# CLI Inference

- Lauch serve cli

        python -m llava.serve.cli \
            --model-path liuhaotian/llava-v1.5-7b \
            --image-file "https://llava-vl.github.io/static/images/view.jpg" \
            --load-4bit

# Train custom dataset

- If you have a limited task-specific data, we recommend finetuning from LLaVA checkpoints with LoRA following this:

        deepspeed llava/train/train_mem.py \
            --lora_enable True \
            --lora_r 128 \
            --lora_alpha 256 \ 
            --mm_projector_lr 2e-5 \
            --deepspeed ./scripts/zero3.json \
            --model_name_or_path liuhaotian/llava-v1.5-13b \
            --version v1 \
            --data_path ./playground/data/llava_v1_5_mix665k.json \
            --image_folder ./playground/data \
            --vision_tower openai/clip-vit-large-patch14-336 \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir ./checkpoints/llava-v1.5-13b-task-lora \
            --num_train_epochs 1 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --learning_rate 2e-4 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb

- If the amount of the task-specific data is sufficient, you can also finetune from LLaVA checkpoints with full-model finetuning following this:


        deepspeed llava/train/train_mem.py \
            --deepspeed ./scripts/zero3.json \
            --model_name_or_path liuhaotian/llava-v1.5-13b \
            --version v1 \
            --data_path ./playground/data/llava_v1_5_mix665k.json \
            --image_folder ./playground/data \
            --vision_tower openai/clip-vit-large-patch14-336 \
            --mm_projector_type mlp2x_gelu \
            --mm_vision_select_layer -2 \
            --mm_use_im_start_end False \
            --mm_use_im_patch_token False \
            --image_aspect_ratio pad \
            --group_by_modality_length True \
            --bf16 True \
            --output_dir ./checkpoints/llava-v1.5-13b-task \
            --num_train_epochs 1 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 1 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50000 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 2048 \
            --gradient_checkpointing True \
            --dataloader_num_workers 4 \
            --lazy_preprocess True \
            --report_to wandb

# References

[llava-repo](https://github.com/haotian-liu/LLaVA)

[llava-docs](https://github.com/haotian-liu/LLaVA/tree/main/docs)

[llava-models_zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)

[llava-colab](https://github.com/camenduru/LLaVA-colab/tree/main)

[llava-v1_5-scripts](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5)

[liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b/tree/main)

[liuhaotian/llava-v1.5-7b-lora](https://huggingface.co/liuhaotian/llava-v1.5-7b-lora/tree/main)

[Finetune_Custom_Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)