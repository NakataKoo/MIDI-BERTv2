import numpy as np
import pickle
from tqdm import tqdm
import utils as utils # import data_creation.prepare_data.utils as utils

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}

class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events(self, input_path, task):
        note_items, tempo_items = utils.read_items(input_path)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        note_items = utils.quantize_items(note_items)
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        return events

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data

    def prepare_data(self, midi_paths, task, max_len):
        all_words, all_ys, midi_ids, not_midis = [], [], [], []
        id = 0
        for path in tqdm(midi_paths):
            # extract events
            events = self.extract_events(path, task)
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                not_midis.append(path)
                continue
            # events to words
            words, ys, midi_id = [], [], []
            for note_tuple in events: # note_tupleは複数のイベントを含むタプルで、音符の高さ、長さ、テンポなどの情報が含まれる
                nts, to_class = [], -1
                for e in note_tuple: # 各イベントeのname（例えば、PitchやDuration）とvalue（例えば、60や120など）を組み合わせてトークンに変換
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text]) # self.event2word[e.name][e_text]で、この文字列をトークン化し、対応する整数IDをntsに追加
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts) # 生成されたトークンのリストntsをwordsリストに追加
                if task == 'melody' or task == 'velocity':
                    ys.append(to_class+1)
            # wordsリストに、MIDIファイル全体のトークン化されたデータが格納されている

            # slice to chunks so that max length = max_len (default: 512)
            slice_words, slice_ys = [], []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len]) # wordsリストの中から、i番目からi+max_len番目までのデータをスライスし、slice_wordsに追加
                
                if task == "composer":
                    name = path.split('/')[-2]
                    slice_ys.append(Composer[name])
                elif task == "emotion":
                    name = path.split('/')[-1].split('_')[0]
                    slice_ys.append(Emotion[name])
                else:
                    slice_ys.append(ys[i:i+max_len])
            
            # padding or drop
            # drop only when the task is 'composer' and the data length < max_len//2
            if len(slice_words[-1]) < max_len:
                if task == 'composer' and len(slice_words[-1]) < max_len//2:
                    slice_words.pop()
                    slice_ys.pop()
                else:
                    slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)

            if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)

            midi_id = [id]*len(slice_words)
            midi_ids = midi_ids + midi_id
            id += 1
            
            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)
        midi_ids = np.array(midi_ids)
        not_midis = np.array(not_midis)

        return all_words, all_ys, midi_ids, not_midis
