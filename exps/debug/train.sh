#!/bin/bash
python train.py train \
    --train=./data/nlie/train.txt \
    --valid=./data/nlie/valid.txt \
    --test=./data/nlie/test.txt \
    --vocab=./data/nlie/vocab.bin \
    --work-dir=./work_dir.debug
