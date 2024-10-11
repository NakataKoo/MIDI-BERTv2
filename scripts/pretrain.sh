export PYTHONPATH="."
python3 MidiBERT/main.py \
--name=20241008_distilbert \
--datasets=lmd_aligned \
--model=distilbert \
--batch_size=32 \
--num_workers=16
