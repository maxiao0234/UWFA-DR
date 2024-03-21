export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM='false'
export FOLD=1
export MODEL_NAME='runwayml/stable-diffusion-v1-5'
export MODEL_ID_OR_ORG='openai/clip-vit-large-patch14'
export DATASET_DIR='datasets/UWFA'
export OUTPUT_DIR='outputs/UWFA'

torchrun train_multi_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--clip_model_id_org=$MODEL_ID_OR_ORG \
--fold=$FOLD \
--train_data_dir=$DATASET_DIR \
--resolution=512 \
--train_batch_size=1 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=20000 \
--checkpointing_steps=20000 \
--learning_rate=1e-04 --lr_scheduler='cosine' --lr_warmup_steps=0 \
--output_dir=$OUTPUT_DIR \
