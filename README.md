# MidiBERT-Piano


事前学習済みモデルの重み＋データセット：https://huggingface.co/KooNakata/MIDI-BERT

## Introduction
This is not official repository for the paper, [MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding](https://arxiv.org/pdf/2107.05223.pdf).

With this repository, you can
* pre-train a MidiBERT-Piano with your customized pre-trained dataset
* fine-tune & evaluate on 4 downstream tasks
* extract melody (mid to mid) using pre-trained MidiBERT-Piano

All the datasets employed in this work are publicly available.

## Installation
* Python3.9
* Install generally used packages for MidiBERT-Piano:
```python
git clone https://github.com/wazenmai/MIDI-BERT.git
cd MIDI-BERT
pip install -r requirements.txt
```

研究室のA40サーバー（CUDA 12.1）では、以下でも上手くいった(torch=2.2.0, CUDA=12.1,)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Python3.10の場合
requirements.txtを以下の様に変更
```
numpy>=1.13.3
matplotlib>=3.3.3
mido==1.2.10
#torch>=1.3.1
chorder==0.1.2
#miditoolkit==0.1.14
#scikit_learn==0.24.2
#torchaudio==0.9.0
transformers==4.8.2
SoundFile
tqdm
pypianoroll
```
次に、以下を実行（CUDA 12.1の場合）
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

miditoolkitやscikit_learnはバージョン指定せず、個別にインストールするのでも良さそう

## Customize your own pre-training dataset 'lmd_aligned'

1. data_creation/prepare_dataのmain.py, model.py, utils.pyのimportにおいて、data_creation.prepare_data.の部分を削除
2. ```!wget http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz```を実行
3. ```!tar -zxvf lmd_aligned.tar.gz```を実行し解凍
4. 以下でdata_creation/prepare_data/dict/CP.pklの中身を表示
```python
import pickle

# CP.pklファイルのパス
file_path = 'data_creation/prepare_data/dict/CP.pkl'

# ファイルを読み込み
with open(file_path, 'rb') as f:
    cp_dict = pickle.load(f)

# データの表示
print(cp_dict)
```
5. 以下でCP.pklにおいて、Pitchの範囲を0~127へ拡大
```python
import pickle

# 既存の辞書を読み込み
dict_path = 'data_creation/prepare_data/dict/CP.pkl'
with open(dict_path, 'rb') as f:
    event2word, word2event = pickle.load(f)

# ピッチの範囲
min_pitch = 0
max_pitch = 127

# ピッチのエントリを追加
for pitch in range(min_pitch, max_pitch + 1):
    pitch_key = f'Pitch {pitch}'
    if pitch_key not in event2word['Pitch']:
        event2word['Pitch'][pitch_key] = -1  # 一時的に-1を設定

# ピッチのキーを昇順にソートして再割り当て
special_keys = {'Pitch <PAD>', 'Pitch <MASK>'}
sorted_pitch_keys = sorted(
    [k for k in event2word['Pitch'].keys() if k not in special_keys],
    key=lambda x: int(x.split()[1])
)

# 特別なキーは元の位置に戻す
for new_index, pitch_key in enumerate(sorted_pitch_keys):
    event2word['Pitch'][pitch_key] = new_index
    word2event['Pitch'][new_index] = pitch_key

# 特別なキーを追加
current_index = len(sorted_pitch_keys)
for special_key in special_keys:
    event2word['Pitch'][special_key] = current_index
    word2event['Pitch'][current_index] = special_key
    current_index += 1

# 更新された辞書を保存
with open(dict_path, 'wb') as f:
    pickle.dump((event2word, word2event), f)

print("CP.pklを更新しました。")
```
6. utils.pyの29行目以降を以下の様に変更
```python
try:
   midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
except OSError as e:
   print(f"Error reading {file_path}: {e}")
   return [], []  # 空のリストを返してエラーを処理
```

7. data_creation/prepare_data/main.pyの120行目あたりを、以下に変更
```python
elif args.input_dir == "lmd_aligned":
    files = glob.glob('lmd_aligned/**/*.mid', recursive=True)
```

8. ルードディレクトリにて以下を実行し、データセットに存在しないファイル＋サブディレクトリをlmd_alignedフォルダから削除
```python
import pandas as pd
import os
import shutil

# CSVファイルを読み込む
df = pd.read_csv('midi_mp3_caption_clean.csv')

# 「lmd_aligned」列に存在するフォルダ名のリストを取得
existing_folders = df['lmd_aligned'].tolist()

# ディレクトリAのパスを指定
directory_a = 'lmd_aligned/'

# ディレクトリA内の一番下の階層のみを走査
for root, dirs, files in os.walk(directory_a):
    if not dirs:  # サブディレクトリがない、つまり一番下の階層である場合
        if root not in existing_folders:
            # 一番下のフォルダが「lmd_aligned」列に存在しない場合、そのフォルダを削除
            shutil.rmtree(root)
            print(f"Deleted folder: {root}")
```

```python
def remove_empty_dirs(directory):
    # ディレクトリ内を再帰的に走査
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # ディレクトリが空かどうかを確認
            if not os.listdir(dir_path):
                # 空のディレクトリを削除
                os.rmdir(dir_path)
                print(f"Deleted empty directory: {dir_path}")

# 対象のディレクトリを指定
directory = 'lmd_aligned'

# 空のサブディレクトリを削除
remove_empty_dirs(directory)
```

以下で、17,077となることを確認
```find lmd_aligned -type d -links 2 | wc -l```

以下で、63,330となることを確認
```!find lmd_aligned -type f | wc -l```

以下で、63,330となることを確認
```python
import glob
files = glob.glob('lmd_aligned/**/*.mid', recursive=True)
len(files)
```

9. not_midis.npyに存在するファイル名を、lmd_alignedから削除
```python
import pandas as pd
import os
import shutil
import numpy as np

# ファイル名を指定してください
file_name = 'not_midis.npy'

# .npyファイルを読み込みます
data = np.load(file_name)
not_folders = data.tolist()

# 特定のディレクトリのパス
target_directory = "lmd_aligned"

# リストAのファイルを削除
for file_path in not_folders:
    # ファイルが存在するか確認
    if os.path.exists(file_path) and file_path.startswith(target_directory):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")
    else:
        print(f"File not found or outside target directory: {file_path}")
```

```python
def remove_empty_dirs(directory):
    # ディレクトリ内を再帰的に走査
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # ディレクトリが空かどうかを確認
            if not os.listdir(dir_path):
                # 空のディレクトリを削除
                os.rmdir(dir_path)
                print(f"Deleted empty directory: {dir_path}")

# 対象のディレクトリを指定
directory = 'lmd_aligned'

# 空のサブディレクトリを削除
remove_empty_dirs(directory)
```

10. not_filesという列をデータセットに追加し、上記で削除したフォルダを記す
```python
import pandas as pd

# CSVファイルをデータフレームに読み込む
df = pd.read_csv('/content/midi_mp3_caption_clean.csv')

# "not_files" 列を追加し、初期値を0に設定
df['not_files'] = 0

# 対象の文字列リストを設定
target_strings = [
    "lmd_aligned/S/E/K/TRSEKRS12903CBB4FF",
    "lmd_aligned/S/B/C/TRSBCXR128F92E1B1B",
    "lmd_aligned/K/Z/Q/TRKZQMX128F14774F1",
    "lmd_aligned/K/Q/Z/TRKQZNL128F92E11A6",
    "lmd_aligned/I/P/N/TRIPNEP128E07877F5",
    "lmd_aligned/I/Q/O/TRIQOWE12903CD02A3",
    "lmd_aligned/L/A/R/TRLARFV128E0793EBF",
    "lmd_aligned/L/M/Q/TRLMQTE12903CC5B9A",
    "lmd_aligned/L/V/T/TRLVTWI128F92EFEA7",
    "lmd_aligned/L/N/O/TRLNOGT128F42971D9",
    "lmd_aligned/L/N/Q/TRLNQGI128F92D5D37",
    "lmd_aligned/R/Y/G/TRRYGTP12903D13BD7",
    "lmd_aligned/R/B/P/TRRBPNS128F9316BB4",
    "lmd_aligned/R/N/C/TRRNCJG12903CFD79A",
    "lmd_aligned/J/T/L/TRJTLSC128F92D28A2",
    "lmd_aligned/J/T/G/TRJTGYX128F4297576",
    "lmd_aligned/J/T/X/TRJTXZB128F42A1018",
    "lmd_aligned/W/C/Q/TRWCQTZ12903CC5BA1",
    "lmd_aligned/W/M/E/TRWMEMW12903CA86D0",
    "lmd_aligned/W/X/W/TRWXWNN128F93112A6",
    "lmd_aligned/W/Q/O/TRWQORX128F42A8E5F",
    "lmd_aligned/Z/N/T/TRZNTLA128EF363651",
    "lmd_aligned/G/J/M/TRGJMRT128F4263548",
    "lmd_aligned/G/Y/W/TRGYWIU128F1468031",
    "lmd_aligned/G/Z/Y/TRGZYUD128F9316BBA",
    "lmd_aligned/D/T/D/TRDTDAT128F426AB1C",
    "lmd_aligned/D/T/Q/TRDTQRI128F42971DB",
    "lmd_aligned/D/T/E/TRDTEGG128F1480E89",
    "lmd_aligned/P/S/K/TRPSKFO12903CA1410",
    "lmd_aligned/P/U/O/TRPUOVN128F92EE1AE",
    "lmd_aligned/O/X/M/TROXMLO128F429757D",
    "lmd_aligned/M/W/E/TRMWEOC128F9322C97",
    "lmd_aligned/M/F/V/TRMFVIY128F9316BB5",
    "lmd_aligned/X/J/B/TRXJBOT128F1496AD4",
    "lmd_aligned/X/O/F/TRXOFFX128F42962D7",
    "lmd_aligned/T/Q/D/TRTQDGL128E0780C94",
    "lmd_aligned/F/I/V/TRFIVUI128F4297586",
    "lmd_aligned/F/Y/B/TRFYBKR128F4297574",
    "lmd_aligned/F/Z/Q/TRFZQXL128F930924A",
    "lmd_aligned/Q/D/F/TRQDFFS128EF363246",
    "lmd_aligned/Q/T/T/TRQTTJS128F9316BAB",
    "lmd_aligned/Q/E/N/TRQENKM128F42A1020",
    "lmd_aligned/V/P/Z/TRVPZQL12903CA4624",
    "lmd_aligned/U/C/G/TRUCGYQ128F9343164",
    "lmd_aligned/U/C/H/TRUCHHA128EF3435EA",
    "lmd_aligned/U/E/U/TRUEUDK128E0782EA2",
    "lmd_aligned/H/D/G/TRHDGDU128F92EA54D",
    "lmd_aligned/B/M/F/TRBMFRN128F92DD39F",
    "lmd_aligned/B/Q/U/TRBQUBO128E0790182",
    "lmd_aligned/N/I/M/TRNIMOG128F4297583",
    "lmd_aligned/N/I/V/TRNIVDJ128F42AAC14",
    "lmd_aligned/N/G/S/TRNGSHX128F42365CE",
    "lmd_aligned/N/X/H/TRNXHWU128F931716F",
    "lmd_aligned/N/E/T/TRNETGG128E079264C"
]

# "lmd_aligned"列の値がtarget_stringsに含まれる場合、"not_files"列の値を1に設定
df.loc[df['lmd_aligned'].isin(target_strings), 'not_files'] = 1

# import ace_tools as tools; tools.display_dataframe_to_user(name="Updated DataFrame", dataframe=df)
```

11. 以下でMIDI-BERT入力用データの前処理実行
```
input_dir="lmd_aligned"
!export PYTHONPATH='.'

# custom directory
!python3 data_creation/prepare_data/main.py --input_dir=$input_dir --name="lmd_aligned"
```

## Citation

[Midi-BERT Official repo](https://github.com/wazenmai/MIDI-BERT/tree/CP).
