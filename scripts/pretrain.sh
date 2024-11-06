export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES="0"
python3 MidiBERT/main.py \
--name=20241031_distilbert \
--datasets=lmd_aligned \
--model=distilbert \
--batch_size=32 \
--num_workers=2 \
--num_hidden_layers=6