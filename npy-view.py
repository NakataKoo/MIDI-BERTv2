import numpy as np

# ファイル名を指定してください
file_name = '/home/Nakata/MIDI-BERT/not_midis.npy'
# .npyファイルを読み込みます
data = np.load(file_name)
print(data)