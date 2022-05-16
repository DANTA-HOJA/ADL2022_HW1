from pathlib import Path
from typing import List, Dict
import json
import copy

import numpy as np

import torch
from torch.utils.data import Dataset

from utils import Vocab

class IntentClsDataset(Dataset):
    def __init__(
        self,
        data_path: Path, # data_path = .json file include {'text':str, 'intent':str, 'id':str}
        vocab: Vocab, # vocab = Vocab(common_words)：幫common_words編號
        label2idx: Dict[str, int], # preprocess 產生的 intent_to_index 檔案的內容 => inten2idx.json
        max_len: int, # 句子最大長度
    ):
        self.data = json.loads(data_path.read_text())
        self.vocab = vocab
        self.label2idx = label2idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        #print(self.data[index])
        
        # text processing 
        sentence = self.data[index]["text"].split() # 把整個句子依單字分割
        sentence_in_index = copy.deepcopy(sentence)
        for i in range(len(sentence)): # 將 "一般句子" 轉換為 "以index表示的句子"
            sentence_in_index[i] =  self.vocab.token_to_id(sentence[i])
        # print(sentence)
        # print(sentence_in_index)
        while len(sentence_in_index) < self.max_len: # 將長度不足的句子padding到設定的最大長度
            sentence_in_index.append(0)
        sentence_in_index = torch.from_numpy(np.array(sentence_in_index)) # 轉為tensor型態
        # print("sentence_in_index：", type(sentence_in_index), f", shape = {sentence_in_index.shape}")
        # print(sentence_in_index)
        
        # intent processing 
        intent = self.data[index]["intent"]
        intent_in_index = self.label2idx[intent] # 將 "intent" 轉換為 "以index表示的intent"
        # print(intent)
        # print(intent_in_index)
        intent_in_index = torch.from_numpy(np.array(intent_in_index)) # 轉為tensor型態
        # print("intent_in_index", type(intent_in_index), f"shape = {intent_in_index.shape}")
        # print(intent_in_index)
        
        return sentence_in_index, intent_in_index

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)

    def label_to_index(self, label: str):
        return self.label2idx[label]

    def index_to_label(self, index: int):
        return self.idx2label[index]
    
class IntentClsDataset_TESTver(Dataset):
    def __init__(
        self,
        data_path: Path, # data_path = .json file include {'text':str, 'intent':str, 'id':str}
        vocab: Vocab, # vocab = Vocab(common_words)：幫common_words編號
        label2idx: Dict[str, int], # preprocess 產生的 intent_to_index 檔案的內容 => inten2idx.json
        max_len: int, # 句子最大長度
    ):
        self.data = json.loads(data_path.read_text())
        self.vocab = vocab
        self.label2idx = label2idx
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        #print(self.data[index])
        
        # text processing 
        sentence = self.data[index]["text"].split() # 把整個句子依單字分割
        sentence_in_index = copy.deepcopy(sentence)
        for i in range(len(sentence)): # 將 "一般句子" 轉換為 "以index表示的句子"
            sentence_in_index[i] =  self.vocab.token_to_id(sentence[i])
        # print(sentence)
        # print(sentence_in_index)
        while len(sentence_in_index) < self.max_len: # 將長度不足的句子padding到設定的最大長度
            sentence_in_index.append(0)
        sentence_in_index = torch.from_numpy(np.array(sentence_in_index)) # 轉為tensor型態
        # print("sentence_in_index：", type(sentence_in_index), f", shape = {sentence_in_index.shape}")
        # print(sentence_in_index)
        
        # id processing
        id_in_str = self.data[index]["id"]
        id_in_int = int(id_in_str.split("-")[1])
        # print(f"id_in_str = {id_in_str}, type(id_in_str) = {type(id_in_str)}")
        # print(f"id_in_int = {id_in_int}, type(id_in_int) = {type(id_in_int)}")
        return sentence_in_index, id_in_int

    @property
    def num_classes(self) -> int:
        return len(self.label2idx)

    def label_to_index(self, label: str):
        return self.label2idx[label]

    def index_to_label(self, index: int):
        return self.idx2label[index]
