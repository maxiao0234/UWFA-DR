export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export EPOCHS=500
export BATCH_SIZE=8
export FOLD=1
export OUTPUT_DIR='outputs/UWFA'
export DATASET_DIR='datasets/UWFA'
export MODEL_ID_OR_ORG='openai/clip-vit-large-patch14'

python train_clip.py \
--epochs=$EPOCHS \
--batch_size=$BATCH_SIZE \
--fold=$FOLD \
--output-dir=$OUTPUT_DIR \
--dataset-dir=$DATASET_DIR \
--model_id_org=$MODEL_ID_OR_ORG \
