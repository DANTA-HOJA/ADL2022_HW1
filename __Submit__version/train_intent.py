import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import os
import math
import time

from tqdm import tqdm
from tqdm import trange
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
    
    # DO: load train and evaluation data for training
    data_paths = {tag: args.data_dir / f"{tag}.json" for tag in TAG_DATASET}
    # print(data_paths) # output => {'train': PosixPath('data/intent/train.json'), 'eval': PosixPath('data/intent/eval.json')}
    train_set = IntentClsDataset(data_paths["train"], vocab, intent2idx, args.max_len)
    eval_set = IntentClsDataset(data_paths["eval"], vocab, intent2idx, args.max_len)
    print(f"\nData in train_set = {train_set.__len__()}, Data in eval_set = {eval_set.__len__()}\n")
    # CHECK_PT: train/eval set load complete
    # input("\n=> train/eval set load complete, press Enter to continue")
    
    # DO: crecate DataLoader for train / dev datasets
    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4),
        'eval': DataLoader(eval_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    }
    print(f"dataloaders = {dataloaders}\n")
    # CHECK_PT: init dataLoader for train/eval set complete
    # input("\n=> init dataLoader for train/eval set complete, press Enter to continue") 
    
    # DO: check if CUDA is available
    if args.device == "cpu":
        print("QQ, you should use a graphic card ~, if you don't have it, just buy it")
    else:
        try:
            gpu_name = torch.cuda.get_device_name(args.device)
        except Exception as e:
            print("*"*100)
            print(f"device {args.device} not found, {e.__class__.__name__}: {e.args[0]}\n") # get detail Error_Info
            if torch.cuda.is_available(): 
                args.device = torch.cuda.current_device()
                print(f"==> find one available gpu at cuda:{args.device}")
                gpu_name = torch.cuda.get_device_name(args.device)
            else:
                args.device = "cpu"
                print("as you know, you don't have a graphic card, stop dreaming ~")
            print("*"*100)
        print(f"\nhi, i am '{gpu_name}' ~")
    device = torch.device(args.device)
    print(f"==> device using {device}\n")
    # DO: clean GPU cache
    if args.device != "cpu":
        torch.cuda.empty_cache() 
        print("torch.cuda.empty_cache()")
    # CHECK_PT: device setting complete
    # input("\n=> device setting complete, press Enter to continue") 
    
    # DO: load embedding
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    embeddings = embeddings.to(device)
    
    # DO: init model
    model = IntentCls_LSTM(embeddings=embeddings, hidden_size=args.hidden_size, num_layers=args.num_layers,
                          dropout= args.dropout, bidirectional= args.bidirectional,
                          num_classes=train_set.num_classes, device=device) # init model
    model = model.to(device) # send model to device
    print(model)
    
    # DO: init optimizer, and loss_function
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # DO: add some parameters
    Epoch_loss_logger = {'train': [], 'eval': []} # loss logger
    Epoch_acc_logger = {'train': [], 'eval': []} # acc logger
    best_avg_loss = 1e10
    best_avg_acc = 0
    
    # DO: Start Training
    epoch_pbar = trange(args.num_epoch)
    for epoch in epoch_pbar:

        # DO: add some parameters
        epoch_weight_ckpt = args.ckpt_dir / f"{model.MODEL_TYPE()}_best_weight.ckpt" # where to save epoch weight file => training recovery
        
        epoch_t_start = time.time() # save start time for a new epoch
        epoch_pbar.desc = f"Epoch: {epoch}, BEST_ACC({best_avg_acc:.2%})"
        
        # DO: Epoch start, one epoch = training + evaluation through all batches
        for TAG in TAG_DATASET:
            
            # DO: add some parameters
            num_batches = 0 # how many batches are generated by dataloader
            # Parameters: loss
            batch_loss = 0 # loss of current batch
            batch_cum_loss = 0 # an accumulation of loss of all batches
            avg_batch_loss = 0 # avg_batch_loss = batch_cum_loss/num_batches
            # Parameters: accuracy
            batch_acc = 0 
            batch_cum_acc = 0
            avg_batch_acc = 0
            
            # DO: set model mode
            if TAG == "train":
                model.train()
            else:
                model.eval()
            
            # DO: Start the train or evaluate process through all batches
            batch_bar = tqdm(dataloaders[TAG], desc=f"{TAG}: ")
            for batch_index, data in enumerate(batch_bar):
                sentence, intent = data # sentence.shape = [batch_size, sentence_max_len]
                                        # intent.shape = [batch_size, 1]             
                sentence, intent = sentence.to(device), intent.to(device)

                torch.set_grad_enabled(False)
                with torch.set_grad_enabled(TAG == "train"): 
                    # DO: predict, calculate loss of this batch, and accumulate the loss
                    pred = model(sentence)# return softmax(IntentCls_RNN) or logsoftmax(IntentCls_LSTM) for each class
                                          # pred.shape = [batch_size, intent_classes]
                    batch_loss = loss_fn(pred, intent) # loss_fn(para:softmax_prob, para:intent_in_index)
                    batch_cum_loss += batch_loss.detach().cpu().numpy() # tensor.detach().cpu().numpy() => copy data from GPU to CPU
                    
                    # DO (train only): update the gradient using backpropagation
                    if TAG == "train":
                        optimizer.zero_grad() # zero the parameter of gradients before backpropagation
                        batch_loss.backward() # use loss to do backpropagation
                        optimizer.step() # update the optimizer with gradient computed in backpropagation
                    
                    # DO: calculate accuracy of this batch, and accumulate the accuracy
                    correct = 0
                    total = intent.shape[0]
                    for i in range(len(pred)): # calculate match item
                        _, pred_cls = torch.max(pred[i],0) # return ["highest probability value" in pred], [index of "highest probability value" in pred]==predicted_intent_class
                        groundtruth_cls = intent[i]
                        if pred_cls == groundtruth_cls: correct+=1
                    batch_acc = correct / total
                    batch_cum_acc += batch_acc
                    
                    # DO: update batch_Info to CMD
                    tqdm.write(f"{TAG}: batch_index = {batch_index}, metrics : batch_accuracy = {batch_acc:.2%} ({correct}/{total}), batch_loss = {batch_loss}")
                
                # CHECK_PT: one batch process complete
                # input(f"\n=> one {TAG} batch process complete, batch_index = {batch_index}, press Enter to continue")
                num_batches += 1
            # end : batch
            os.system("clear")
            
            # DO: calculate the "average loss" in "train/eval set", and add to logger
            avg_batch_loss = batch_cum_loss/num_batches
            #print(type(avg_batch_loss), type(Epoch_loss_logger[TAG]))
            Epoch_loss_logger[TAG].append(avg_batch_loss)
            
            # DO: calculate the "average accuracy" in "train/eval set", and add to logger
            avg_batch_acc = batch_cum_acc/num_batches
            #print(type(avg_batch_loss), type(Epoch_loss_logger[TAG]))
            Epoch_acc_logger[TAG].append(avg_batch_acc)
            
            # DO: update Info to CMD
            tqdm.write(f"Epoch_loss_logger: train_len = {len(Epoch_loss_logger['train'])}, eval_len = {len(Epoch_loss_logger['eval'])}")
            tqdm.write("="*100)
            tqdm.write("") # # write empty new_line
            tqdm.write(f"{TAG}: number of batches = {num_batches}, avg_batch_acc = {avg_batch_acc:.2%}, avg_batch_loss = {avg_batch_loss}")

            # DO: save the best model info
            if TAG == 'eval' and avg_batch_loss < best_avg_loss and avg_batch_acc > best_avg_acc:
                best_avg_loss = avg_batch_loss
                best_avg_acc = avg_batch_acc
                best_weight_ckpt = args.ckpt_dir / f"{model.MODEL_TYPE()}_best_weight.ckpt" # where to save best weight file
                tqdm.write(f"\nsaving best model to {best_weight_ckpt}\n")
                torch.save({
                            # Info: model_state_dict, optimizer_state_dict(discard)
                            'model_state_dict': model.state_dict(),
                            #'optimizer_state_dict': optimizer.state_dict(),
                            # Info: Epoch_loss_logger, best_avg_loss
                            'trian_loss': Epoch_loss_logger['train'],
                            'eval_loss' : Epoch_loss_logger['eval'],
                            'best_avg_loss': best_avg_loss,
                            # Info: Epoch_acc_logger, best_avg_acc
                            'trian_acc': Epoch_acc_logger['train'],
                            'eval_acc' : Epoch_acc_logger['eval'],
                            'best_avg_acc': best_avg_acc,
                            # Info: epoch, total_epoch(manual)
                            'epoch': [epoch+1, args.num_epoch],
                            'batch_size': args.batch_size,
                            'model_para':{
                                'hidden_size' : args.hidden_size,
                                'num_layers' : args.num_layers,
                                'dropout' : args.dropout,
                                'bidirectional' : args.bidirectional,
                                'num_classes' : train_set.num_classes
                                }
                            }, best_weight_ckpt)
            
            # CHECK_PT: all batches process complete
            # input(f"\n=> all {TAG} batches process complete, press Enter to continue")
        # end : epoch
        
        # DO: calculate the consuming time of this epoch
        epoch_t_stop = time.time() # update the stop time for this epoch
        cost_min = (epoch_t_stop - epoch_t_start) / 60 
        cost_sec = (epoch_t_stop - epoch_t_start) % 60
        
        # DO: update Info to CMD
        tqdm.write(f"epoch = {epoch}, time_cost = {math.floor(cost_min):.0f} m {math.floor(cost_sec):.0f} s, best_avg_acc = {best_avg_acc:.2%}, best_avg_loss = {best_avg_loss}")
        tqdm.write("") # write empty new_line
        tqdm.write("="*100)
        
        # DO: draw and save graph
        # 1. loss graph
        plt.figure("Epoch_loss")
        plt.title("Epoch_loss")
        plt.plot(Epoch_loss_logger['train'], label="train")
        plt.plot(Epoch_loss_logger['eval'], label="eval")
        plt.legend()
        plt.savefig("epoch_Loss_logger.png")
        plt.close()
        # 2. accuracy graph
        plt.figure("Epoch_acc")
        plt.title("Epoch_acc")
        plt.plot(Epoch_acc_logger['train'], label="train")
        plt.plot(Epoch_acc_logger['eval'], label="eval")
        plt.legend()
        plt.savefig("epoch_Acc_logger.png")
        plt.close()
        
        # CHECK_PT: one epoch process complete
        # input(f"\n=> one epoch process complete, press Enter to continue")
    # CHECK_PT: all epochs process complete
    # input(f"\n=> all epochs process complete, press Enter to continue")
    
    # DO: release GPU cache
    if args.device != "cpu":
        torch.cuda.empty_cache()
        print("torch.cuda.empty_cache()")
    # CHECK_PT: release GPU cache complete
    # input(f"\n=> release GPU cache complete, press Enter to continue")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset. (file_extension = .json)",
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
        help="Directory to save the model_checkpoint file. (file_extension = .ckpt)",
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
        "--device", type=str, help="cpu, cuda, cuda:{device_num}, e.g.cuda:0", default="cuda:1"
    )
    parser.add_argument("--num_epoch", type=int, default=30)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
