import json
import os
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import whisper
from transformers import WhisperTokenizer


random.seed(1)

class LibriDataset(Dataset):
    def __init__(self, path, loadtarget=True, tokenizer=None, biasing=True):
        with open(path) as f:
            self.data = json.load(f)
        self.data_idx = list(self.data.keys())
        self.loadtarget = loadtarget
        self.tokenizer = tokenizer
        self.biasing = biasing
        # self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base.en", language="en", task="transcribe")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        uttname = self.data_idx[index]
        data = self.data[uttname]
        data_path = data["fbank"]
        fbank = torch.load(data_path)
        target = data["words"].lower()
        tokenized_words = []
        if self.loadtarget and self.tokenizer is not None:
            # Fake capitalise biasing words
            if self.biasing:
                targetwords = []
                for word in target.split():
                    if word.upper() in data["blist"] and random.random() > 0.5:
                        targetwords.append(word.upper()[0:1]+word[1:])
                    else:
                        targetwords.append(word)
                target = " ".join(targetwords)
            target = self.tokenizer.encode(" "+target) + [self.tokenizer.tokenizer.eos_token_id]
        if self.biasing:
            tokenized_words = []
            for word in data["blist"]:
                word = word.lower()
                wordcap = word[0:1].upper() + word[1:]
                tok_word = self.tokenizer.encode(" " + word)
                tokenized_words.append(tuple(tok_word))
                tokenized_words.append(tuple(self.tokenizer.encode(" "+wordcap)))
        elif self.loadtarget:
            raise Exception("No tokenizer provided to dataloader")
        return uttname, fbank, target, tokenized_words


def check_in_utt(tok_word, target):
    for i in range(len(target)):
        if target[i:i+len(tok_word)] == tok_word:
            return True
    return False


def make_lexical_tree(word_dict, subword_dict, word_unk):
    # node [dict(subword_id -> node), word_id, word_set[start-1, end]]
    root = [{}, -1, None]
    for w, wid in word_dict.items():
        if wid > 0 and wid != word_unk:  # skip <blank> and <unk>
            if True in [c not in subword_dict for c in w]:  # skip unknown subword
                continue
            succ = root[0]  # get successors from root node
            for i, c in enumerate(w):
                cid = subword_dict[c]
                if cid not in succ:  # if next node does not exist, make a new node
                    succ[cid] = [{}, -1, (wid - 1, wid)]
                else:
                    prev = succ[cid][2]
                    succ[cid][2] = (min(prev[0], wid - 1), max(prev[1], wid))
                if i == len(w) - 1:  # if word end, set word id
                    succ[cid][1] = wid
                succ = succ[cid][0]  # move to the child successors
    return root


def collate_wrapper(batch):
    uttnames = [i[0] for i in batch]
    fbank = torch.stack([i[1] for i in batch])
    tgt = [i[2] for i in batch]
    blist = []
    for i in batch:
        for word in i[3]:
            if word not in blist:
                blist.append(word)
    return uttnames, fbank, tgt, blist


def get_dataloader(path, bs, shuffle=True, loadtarget=True, tokenizer=None, biasing=False):
    dataset = LibriDataset(path, loadtarget=loadtarget, tokenizer=tokenizer, biasing=biasing)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        pin_memory=True,
    )


class BiasingProcessor(object):
    def __init__(self, tokenizer, fulllist, ndistractors=500, drop=0.3):
        self.all_rare_words = []
        with open(fulllist) as fin:
            for line in fin:
                word = line.lower().strip()
                tok_word = tokenizer.encode(' '+word)
                self.all_rare_words.append(tuple(tok_word))
                wordcap = word[0:1].upper() + word[1:]
                self.all_rare_words.append(tuple(tokenizer.encode(' '+wordcap)))
        self.ndistractors = ndistractors
        self.drop = drop
        self.chardict = {idx:idx for idx in range(tokenizer.tokenizer.vocab_size)}

    def insert_distractors(self, uttblist):
        if self.drop > 0:
            uttblist = random.sample(uttblist, int(len(uttblist) * (1 - self.drop)))
        uttblist = [tuple(bword) for bword in uttblist]
        pool = random.sample(self.all_rare_words, self.ndistractors)
        for word in pool:
            if word not in uttblist:
                uttblist.append(word)
        uttblist = uttblist[:self.ndistractors]
        return uttblist

    def construct_tree(self, uttblist):
        worddict = {word: i+1 for i, word in enumerate(uttblist)}
        lextree = make_lexical_tree(worddict, self.chardict, -1)
        return lextree

    def get_lextree(self, uttblist):
        uttblist = self.insert_distractors(uttblist)
        lextree = self.construct_tree(uttblist)
        return lextree
