#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

#NUM_GPU=4

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
#PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
python train.py \
    --model_name_or_path bert-large-uncased \
    --train_file data/nli_for_simcse.csv \
    --output_dir result/my-sup-simcse-bert-large-uncased-1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 256 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 50 \
    --pooler_type avg \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    "$@"
