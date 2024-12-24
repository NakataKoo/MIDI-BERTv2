export PYTHONPATH="."
export CUDA_VISIBLE_DEVICES="0"
python3 MidiBERT/main.py \
--name=20241224_distilbert \
--datasets=lmd_aligned \
--model=distilbert \
--batch_size=32 \
--num_workers=4 \
--max_seq_len=1024