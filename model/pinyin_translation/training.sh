#!/bin/bash

source init.sh

## training
CUDA_VISIBLE_DEVICES=0 fairseq-train ${data_dir}/data-bin --arch transformer_iwslt_de_en \
    --source-lang ${src} --target-lang ${tgt}  \
    --optimizer adam  --lr 5e-4 --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3  --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1  --warmup-updates 4000 \
    --keep-last-epochs 1 --num-workers 4 --max-epoch 20 \
    --save-dir ${model_dir}/checkpoints
    