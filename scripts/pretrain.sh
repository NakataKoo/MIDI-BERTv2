export PYTHONPATH="."
python3 MidiBERT/main.py \
--name=20241008_bert \
--datasets=lmd_aligned \
--model=bert \
--batch_size=64 \
--num_workers=16 \
--num_hidden_layers=6
