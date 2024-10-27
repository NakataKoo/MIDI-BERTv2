export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES="0, 1"
python3 MidiBERT/main.py \
--name=20241027_albert \
--datasets=lmd_aligned \
--model=albert \
--batch_size=32 \
--num_workers=2 \
--num_hidden_layers=6