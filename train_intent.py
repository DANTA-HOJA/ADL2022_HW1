import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import os
from tqdm import tqdm
from tqdm import trange
import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn

from utils import Vocab
from model import IntentCls_RNN, IntentCls_LSTM
from dataset import IntentClsDataset

TAG_DATASET = ["train", "eval"]

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f) # Vocab(common_words)

    intent2idx_json = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent2idx_json.read_text())

    data_paths = {tag: args.data_dir / f"{tag}.json" for tag in TAG_DATASET}
    # print(data_paths) # {'train': PosixPath('data/intent/train.json'), 'eval': PosixPath('data/intent/eval.json')}

    train_set = IntentClsDataset(data_paths["train"], vocab, intent2idx, args.max_len)
    eval_set = IntentClsDataset(data_paths["eval"], vocab, intent2idx, args.max_len)
    print(f"\nData in train_set = {train_set.__len__()}, Data in train_set = {eval_set.__len__()}")
    input("\n=> press Enter to continue")
    
    # DO: crecate DataLoader for train / dev datasets
    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'eval': DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    }
    print(f"\ndataloaders = {dataloaders}")
    input("\n=> press Enter to continue\n") 
    
    # DO: check if CUDA is available
    if not torch.cuda.is_available():
        print("QQ, you should buy a graphic card ~, device chang to 'cpu'\n")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f'==> device using {device}')
    
    # DO: load embedding
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    embeddings = embeddings.to(device)
    
    # DO: init model
    model = IntentCls_RNN(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                          dropout= args.dropout, bidirectional= args.bidirectional,
                          num_classes=train_set.num_classes, device=device) # init model
    model = model.to(device) # send model to device
    print(model)
    
    # DO: init optimizer, and loss_function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # DO: add some parameters
    best_weight_ckpt = args.ckpt_dir / "best_weight.ckpt" # where to save file
    best_loss = 1e10
    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        
        # DO: add some parameters
        Dataset_loss_logger = {'train': [], 'eval': []} # logger
        epoch_since = time.time() # save start time for new epoch
        
        # DO: Each epoch has a training and validation phase
        for TAG in TAG_DATASET:
            
            Dataset_loss = 0
            batches_in_Dataset = 0
            if TAG == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # DO: Start the training process through all batches
            batch_bar = tqdm(dataloaders[TAG], desc=TAG)
            for batch_index, data in enumerate(batch_bar):
                sentence, intent = data
                sentence, intent = sentence.to(device), intent.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(TAG == 'train'):
                    pred = model(sentence)
                    # pred_intent = [] # store: softmax probability to pre_intent
                    for i in range(len(pred)): # len(pred) = [batch_size=64] sentences
                        # # each num in pred["x"th] is the probability of each class[classes=150]
                        # max_prob=0
                        # pred_cls=0
                        # for j in range(len(pred[i])): # len(pred[i]) = [classes=150] probabilities 
                        #     curr_cls_prob = pred[i][j]
                        #     if max_prob < curr_cls_prob:
                        #         max_prob = curr_cls_prob
                        #         pred_cls = j
                        # pred_intent.append(pred_cls)
                        # tqdm.write(f"cls_prob = {max_prob},pred_cls = {pred_cls}")
                        cls_prob_torchMax, pred_cls_torchMax = torch.max(pred[i],0)
                        #tqdm.write(f"using torch.max(), cls_prob = {cls_prob_torchMax},pred_cls = {pred_cls_torchMax}")
                        groundtruth_cls = intent[i]
                        #tqdm.write(f"groundtruth_cls = {groundtruth_cls}")
                    # pred_intent = torch.from_numpy(np.array(pred_intent)) # list(pred_intent) -> tensor(pred_intent)
                    batch_loss = loss_fn(pred, intent) # loss_fn(softmax_prob, intent_in_index)
                    tqdm.write(f"batch_index = {batch_index}, metrics : batch_loss in {TAG}_set = {batch_loss}")
                    Dataset_loss += batch_loss.detach().cpu().numpy()
                    
                    if TAG == "eval":
                        correct = 0
                        # DO: calculate match item
                        for i in range(len(pred)):
                            _, pred_cls = torch.max(pred[i],0)
                            groundtruth_cls = intent[i]
                            if pred_cls == groundtruth_cls: correct+=1
                        
                        tqdm.write(f"batch_index = {batch_index}, metrics : batch_loss in {TAG}_set = {batch_loss}")
                        print(pred_cls)
                        print(intent)
                        print(correct)
                        input()
                    
                    # backward + optimize only if in training phase
                    if TAG == 'train':
                        batch_loss.backward()
                        optimizer.step()
                batches_in_Dataset += 1
            # end : batch
            
            # DO: calculate the avg_loss in Train/eval Dataset, and add the value to logger
            avg_Dataset_loss = Dataset_loss/batches_in_Dataset
            Dataset_loss_logger[TAG].append(avg_Dataset_loss)
            tqdm.write(f"batches_in_Dataset = {batches_in_Dataset}, avg_Dataset_loss = {avg_Dataset_loss}")
            #input(f"\n=> end of {TAG}, press Enter to continue")

            # DO: save the best model info
            if TAG == 'eval' and avg_Dataset_loss < best_loss:
                if avg_Dataset_loss < best_loss:
                    tqdm.write(f"saving best model to {best_weight_ckpt}")
                    best_loss = avg_Dataset_loss
                    torch.save({
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'trian_loss': Dataset_loss_logger['train'],
                                'eval_loss' : Dataset_loss_logger['eval']
                                }, best_weight_ckpt)  
        
        
        # DO: calculate the consuming time of this epoch
        epoch_time_elapsed = time.time() - epoch_since # update stop time
        cost_min = epoch_time_elapsed / 60 
        cost_sec = epoch_time_elapsed % 60
        
        # DO: update Info to CMD
        tqdm.write(f"epoch = {epoch}, {cost_min}m {cost_sec}s, best_loss = {best_loss}")

        # logs graph
        if epoch%10 == 0:
            plt.plot(Dataset_loss_logger['train'])
            plt.plot(Dataset_loss_logger['eval'])
            plt.legend(['train', 'eval'])
            plt.title('loss')
            plt.show()

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    # training
    parser.add_argument(
        "--device", type=str, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
