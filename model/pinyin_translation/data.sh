#!/bin/bash

source init.sh

## tool preparation
git clone https://github.com/moses-smt/mosesdecoder.git
git clone https://github.com/rsennrich/subword-nmt.git


## bpe
python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/train.raw.en  -s 32000 -o ${model_dir}/bpecode.en --write-vocabulary ${model_dir}/voc.en
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.en --vocabulary ${model_dir}/voc.en < ${data_dir}/train.raw.en > ${data_dir}/train.bpe.en
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.en --vocabulary ${model_dir}/voc.en < ${data_dir}/valid.raw.en > ${data_dir}/valid.bpe.en

python ${BPEROOT}/learn_joint_bpe_and_vocab.py --input ${data_dir}/train.raw.zh  -s 32000 -o ${model_dir}/bpecode.zh --write-vocabulary ${model_dir}/voc.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/train.raw.zh > ${data_dir}/train.bpe.zh
python ${BPEROOT}/apply_bpe.py -c ${model_dir}/bpecode.zh --vocabulary ${model_dir}/voc.zh < ${data_dir}/valid.raw.zh > ${data_dir}/valid.bpe.zh


## data cleaning
${CLEAN} ${data_dir}/train.bpe zh en ${data_dir}/train 1 256
${CLEAN} ${data_dir}/valid.bpe zh en ${data_dir}/valid 1 256


## fairseq-preprocess
fairseq-preprocess --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${data_dir}/train --validpref ${data_dir}/valid \
    --destdir ${data_dir}/data-bin
    