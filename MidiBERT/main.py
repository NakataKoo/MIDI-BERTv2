import argparse
import numpy as np
import random
import pickle
import os
import json
import torch

from torch.utils.data import DataLoader
from transformers import BertConfig, AlbertConfig, RobertaConfig, DistilBertConfig
from model import MidiBert
from trainer import BERTTrainer
from midi_dataset import MidiDataset


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--name', type=str, default='MidiBert')

    ### pre-train dataset ###
    parser.add_argument("--datasets", type=str, nargs='+', default=['pop909','composer', 'pop1k7', 'ASAP', 'emopia'])
    
    ### model
    parser.add_argument("--model", type=str, default="bert")

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mask_percent', type=float, default=0.15, help="Up to `valid_seq_len * target_max_percent` tokens will be masked out for prediction")
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)      # hidden state
    parser.add_argument('--epochs', type=int, default=250, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate')

    ### ckpt
    parser.add_argument('--load_ckpt', type=str, default="False", help='load ckpt and restart to train')
    parser.add_argument('--ckpt', type=str, default="", help='ckpt path')
    
    ### cuda ###
    parser.add_argument("--cpu", action="store_true")   # default: False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0, 1], help="CUDA device ids")

    args = parser.parse_args()

    return args


def load_data(datasets):
    to_concat = []
    root = 'Data/CP_data'

    for dataset in datasets:
        if dataset == 'lmd_aligned':
            data = np.load(os.path.join(root, f'{dataset}.npy'), allow_pickle=True)

        print(f'   {dataset}: {data.shape}')
        to_concat.append(data)


    training_data = np.vstack(to_concat)
    print('   > all training data:', training_data.shape)
    
    # shuffle during training phase
    index = np.arange(len(training_data))
    np.random.shuffle(index)
    training_data = training_data[index]
    split = int(len(training_data)*0.85)
    X_train, X_val = training_data[:split], training_data[split:]
    
    return X_train, X_val


def main():
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset", args.datasets) 
    X_train, X_val = load_data(args.datasets)
    
    trainset = MidiDataset(X=X_train)
    validset = MidiDataset(X=X_val) 

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    print("   len of train_loader",len(train_loader))
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
    print("   len of valid_loader",len(valid_loader))

    if args.model == 'bert':
        print("\nBuilding BERT model")
        configuration = BertConfig(max_position_embeddings=args.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=args.hs, # 768
                                    attn_implementation="eager",
                                    vocab_size = 800
        )
    elif args.model == 'albert':
        print("\nBuilding ALBERT model")
        configuration = AlbertConfig(max_position_embeddings=args.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=args.hs, # 768
                                    embedding_size=128,
                                    attn_implementation="eager",
                                    vocab_size = 800
        )
    elif args.model == 'roberta':
        configuration = RobertaConfig(max_position_embeddings=args.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=args.hs, # 768
                                    vocab_size = config.midi.vocab_size
        )
    elif args.model == 'distilbert':
        print("\nBuilding DistilBERT model")
        configuration = DistilBertConfig(max_position_embeddings=args.max_seq_len, # 512
                                    position_embedding_type='relative_key_query',
                                    hidden_size=args.hs, # 768
                                    vocab_size = 800
        )
    midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e, model_name=args.model)
    
    stdict_o = None
    print("Load Checkpoint?: "+args.load_ckpt)
    print(args.ckpt)
    if args.load_ckpt=="True":
        print("\nLoad Check point to restart")
        cpt = torch.load(args.ckpt)
        stdict_m = cpt['state_dict']
        stdict_o = cpt['optimizer']
        midibert.load_state_dict(stdict_m, strict=False)

    print("\nCreating BERT Trainer")
    trainer = BERTTrainer(midibert, train_loader, valid_loader, args.lr, args.batch_size, args.max_seq_len, args.mask_percent, args.cpu, args.cuda_devices, 
                        args.load_ckpt, 
                        stdict_o)
    
    print("\nTraining Start")
    save_dir = 'MidiBERT/result/pretrain/' + args.name
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, 'model.ckpt')
    print("   save model at {}".format(filename))

    best_acc, best_epoch = 0, 0
    bad_cnt = 0

    for epoch in range(args.epochs):
        if bad_cnt >= 30:
            print('valid acc not improving for 30 epochs')
            break
        train_loss, train_acc = trainer.train()
        valid_loss, valid_acc = trainer.valid()

        weighted_score = [x*y for (x,y) in zip(valid_acc, midibert.n_tokens)]
        avg_acc = sum(weighted_score)/sum(midibert.n_tokens)
        
        is_best = avg_acc > best_acc
        best_acc = max(avg_acc, best_acc)
        
        if is_best:
            bad_cnt, best_epoch = 0, epoch
        else:
            bad_cnt += 1
        
        print('epoch: {}/{} | Train Loss: {} | Train acc: {} | Valid Loss: {} | Valid acc: {}'.format(
            epoch+1, args.epochs, train_loss, train_acc, valid_loss, valid_acc))

        trainer.save_checkpoint(epoch, best_acc, valid_acc, 
                                valid_loss, train_loss, is_best, filename)


        with open(os.path.join(save_dir, 'log'), 'a') as outfile:
            outfile.write('Epoch {}: train_loss={}, train_acc={}, valid_loss={}, valid_acc={}\n'.format(
                epoch+1, train_loss, train_acc, valid_loss, valid_acc))


if __name__ == '__main__':
    main()
