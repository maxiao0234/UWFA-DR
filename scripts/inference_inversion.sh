export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM='false'
export FOLD=1
export DATASET_DIR='datasets/UWFA'
export OUTPUT_DIR='outputs/UWFA'
export RESULT_DIR='results/UWFA'
export MODEL_NAME='runwayml/stable-diffusion-v1-5'
export MODEL_ID_OR_ORG='/home/data/maxiao/HuggingFace/models/openai/clip-vit-large-patch14'

python inference_inversion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--clip_model_id_org=$MODEL_ID_OR_ORG \
--fold=$FOLD \
--dataset-dir=$DATASET_DIR \
--output-dir=$OUTPUT_DIR \
--result-dir=$RESULT_DIR \
