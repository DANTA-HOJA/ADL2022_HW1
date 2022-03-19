from itertools import count
import json
import logging
from os import system
import pickle
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from random import random, seed
from time import process_time_ns
from typing import List, Dict # 型別註釋(type annotations)讓開發者或協作者可以更加了解某個變數的型別，
                              # 也讓第三方的工具能夠實作型別檢查器(type checker)

import torch
from tqdm.auto import tqdm

from utils import Vocab

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def build_vocab(words: Counter, vocab_size: int, output_dir: Path, glove_path: Path) -> None:
    common_words = {w for w, _ in words.most_common(vocab_size)}
    # print(f"type(common_words) = {type(common_words)}, \
        # len(common_words) = {len(common_words)}, common_words.element = \n{common_words}") ## -----------------------------------------------------------------testing：印出來了解內容
    # input() ## -----------------------------------------------------------------testing：模擬system("pause")
    # system("clear") ## -----------------------------------------------------------------testing：system("clear")
    
    vocab = Vocab(common_words)
    vocab_path = output_dir / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logging.info(f"Vocab saved at {str(vocab_path.resolve())}")

    glove: Dict[str, List[float]] = {}
    logging.info(f"Loading glove: {str(glove_path.resolve())}")
    with open(glove_path) as fp:
        row1 = fp.readline()
        # if the first row is not header
        if not re.match("^[0-9]+ [0-9]+$", row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in tqdm(enumerate(fp)):
            # print(f"i = {i}, type(line) = {type(line)}, len(line) = {len(line)}, line = \n{line}")
            # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")

            cols = line.rstrip().split(" ")
            # print(f"\n\ntype(glove) = {type(cols)}, len(glove) = {len(cols)}, glove = \n{cols}") ## -----------------------------------------------------------------testing：印出來了解內容
            # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")\

            word = cols[0]
            # print(f"\n\ntype(word) = {type(word)}, len(word) = {len(word)}, word = {word}") ## -----------------------------------------------------------------testing：印出來了解內容
            # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")\

            vector = [float(v) for v in cols[1:]]
            # print(f"type(vector) = {type(vector)}, len(vector) = {len(vector)}, vector = \n{vector}") ## -----------------------------------------------------------------testing：印出來了解內容
            # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")\
            # system("clear") ## -----------------------------------------------------------------testing：system("clear")

            # skip word not in words if words are provided
            if word not in common_words: # word => 所有在 glove 裡的所有字, common_words => 蒐集"train.json"及"eval.json"
                                         # 裡的所有單字後產生words，取words.most_common()的結果
                continue
            glove[word] = vector
            glove_dim = len(vector)

    assert all(len(v) == glove_dim for v in glove.values())
    assert len(glove) <= vocab_size

    num_matched = sum([token in glove for token in vocab.tokens]) # vocab = Vocab(common_words)
    logging.info(
        f"Token covered: {num_matched} / {len(vocab.tokens)} = {num_matched / len(vocab.tokens)}"
    )
    embeddings: List[List[float]] = [
        glove.get(token, [random() * 2 - 1 for _ in range(glove_dim)])
        # Dict[].get(label, value) => 如果label存在Dict[]中 return label的value，不存在則回傳指定value
        # 如果glove找的到從common_words建立的token，則將token對應的vector存入embeddings中，
        # 如果glove找不到從common_words建立的token，則將利用random()隨機產生一個300D的vector存入embeddings中。
        for token in vocab.tokens
    ]
    #print(embeddings) ## -----------------------------------------------------------------testing：印出來了解內容
    embeddings = torch.tensor(embeddings)
    embedding_path = output_dir / "embeddings.pt"
    torch.save(embeddings, str(embedding_path))
    logging.info(f"Embedding shape: {embeddings.shape}") 
    # torch.Size([6491, 300]) => total 6491 words, each word embedding to one 300 for dimension vector
    # index_num of 6491 words refer to Vocab(common_words).token2idx[word]
    logging.info(f"Embedding saved at {str(embedding_path.resolve())}")


def main(args):
    seed(args.rand_seed)

    intents = set()
    words = Counter()
    for split in ["train", "eval"]:
        dataset_path = args.data_dir / f"{split}.json"
        dataset = json.loads(dataset_path.read_text())
        logging.info(f"Dataset loaded at {str(dataset_path.resolve())}")
        # print(f"type(dataset) = {type(dataset)}\n") ## -----------------------------------------------------------------testing：印出來了解內容
        # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")
        # print(f"dataset = \n{dataset[:1]}\n\n")
        # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")
        
        intents.update({instance["intent"] for instance in dataset}) #update()：添加新的元素或集合到當前集合中，如果添加的元素在集合中已存在，
                                                                     #          則該元素只會出現一次，重複的會忽略。
        words.update(
            [token for instance in dataset for token in instance["text"].split()] #split()沒傳入引數 = 以空格為分隔符，包含'\n'
        )
        """# 以下等同於上方 words.update() 的操作
        for instance in dataset:
            print(f"\ninstance in dataset = {instance}") ## -----------------------------------------------------------------testing：印出來了解內容
            for token in instance["text"].split():
                print(f"\ntoken in instance[\"text\"].split() = {token}") ## -----------------------------------------------------------------testing：印出來了解內容
                words2.update([token]) ## 用 [token] 把 token 當成一個字裝，否則會裝到單個字母
        print(f"\n\n\nlen(words) = {len(words)}\nwords = {words.elements}") ## -----------------------------------------------------------------testing：印出來了解內容
        print(f"\n\n\nlen(words2) = {len(words2)}\nwords2 = {words2.elements}") ## -----------------------------------------------------------------testing：印出來了解內容
        logging.info(f"words == words2 ?? => {words==words2}") ## -----------------------------------------------------------------testing：印出來了解內容
        # input("press Enter to continue")
        """
    # 收錄完 "train.json", "eval.json" 後的結果，in this case => total words = 6489, total intents = 150（0-149）
    # print(f"total len(words) = {len(words)}\ntotal words = {words.elements}") ## -----------------------------------------------------------------testing：印出來了解內容
    # print(f"\n\n\nlen(intents) = {len(intents)}\nintents = {intents}") ## -----------------------------------------------------------------testing：印出來了解內容
    # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")
    # system("clear") ## -----------------------------------------------------------------testing：system("clear")

    
    intent2idx = {tag: i for i, tag in enumerate(intents)} # enumerate(Iterable[variable]) -> i（auto index）, element in  Iterable[variable]
    intent_tag_path = args.output_dir / "intent2idx.json"
    # json.dumps()用於將dict型別的資料轉成str，因為如果直接將dict型別的資料寫入json檔案中會發生報錯，
    # 因此在將資料寫入時需要用到該函式。↓
    intent_tag_path.write_text(json.dumps(intent2idx, indent=2)) # indent：沒啥用，就是幫output的.json縮排，數字越大縮越多
    logging.info(f"Intent 2 index saved at {str(intent_tag_path.resolve())}")
    # input("press Enter to continue") ## -----------------------------------------------------------------testing：模擬system("pause")
    
    build_vocab(words, args.vocab_size, args.output_dir, args.glove_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--glove_path",
        type=Path,
        help="Path to Glove Embedding.",
        default="./glove.840B.300d.txt",
    )
    parser.add_argument("--rand_seed", type=int, help="Random seed.", default=13)
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the processed file.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="Number of token in the vocabulary",
        default=10_000, # 可以用下底線任意標記數字增加易讀性，不影響編譯器解讀，10_000 = 10000
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
