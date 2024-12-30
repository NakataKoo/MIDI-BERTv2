export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES="12, 13, 14, 15"
python3 MidiBERT/main.py \
--name=20241230_roberta \
--datasets=lmd_aligned \
--model=roberta \
--batch_size=32 \
--num_workers=16