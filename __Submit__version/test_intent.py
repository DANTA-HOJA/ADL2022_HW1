import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import time
import math

from tqdm import tqdm
import csv

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils import Vocab
from model import IntentCls_RNN, IntentCls_LSTM
from dataset import IntentClsDataset, IntentClsDataset_TESTver

def write_csv(csv_out_path, pred_result: List[Dict]):
    
    f = csv.writer(open(csv_out_path, "w", newline=''))

    # Write CSV Header, If you dont need that, remove this line
    f.writerow(["id", "intent"])

    for item in pred_result:
        f.writerow([item["id"], item["intent"]])
        

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f) # Vocab(common_words)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    # args.test_file = "./data/intent/test.json" 
    test_path = args.test_file
    test_set = IntentClsDataset_TESTver(test_path, vocab, intent2idx, args.max_len)
    print(f"\nData in test_set = {test_set.__len__()}")
    # print(f"\nData in test_set = {test_set.__getitem__(22)}")
    # CHECK_PT: test_set load complete
    # input("\n=> test_load complete, press Enter to continue")
    
    # DO: crecate DataLoader for test dataset
    dataloaders = {"test": DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)}
    print(f"dataloaders = {dataloaders}\n")
    # CHECK_PT: init dataLoader for test_set complete
    # input("\n=> init dataLoader for test_set complete, press Enter to continue") 

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
    # CHECK_PT: device setting complete
    # input("\n=> device setting complete, press Enter to continue") 

    # DO: load .ckpt file to recovery model structure
    # 1. load full stored file = model+training_info
    ckpt_path = args.ckpt_path # .ckpt file path
    training_info = torch.load(ckpt_path) # load .ckpt file
    # 2. load embedding
    embeddings = torch.load("./cache/intent/embeddings.pt")
    embeddings = embeddings.to(device)
    # 3. load "model parameters"
    #batch_size = training_info["batch_size"]
    hidden_size = training_info["model_para"]["hidden_size"]
    num_layers = training_info["model_para"]["num_layers"]
    dropout = training_info["model_para"]["dropout"]
    bidirectional = training_info["model_para"]["bidirectional"]
    num_classes = training_info["model_para"]["num_classes"]
    # 4. create same model structure store in .ckpt file
    # ==> because we only save model gradient (using "model.state_dict()") not whole "model info(include "model structure")"
    model = IntentCls_LSTM(embeddings=embeddings, hidden_size=hidden_size, num_layers=num_layers,
                            dropout= dropout, bidirectional= bidirectional,
                            num_classes=num_classes, device=device) # init model
    model.load_state_dict(training_info['model_state_dict']) # load model 
    model = model.to(device) # send model to device
    print(f"{model}\n")
    #print(f"batch_size = {batch_size}")
    print(f"hidden_size = {hidden_size}")
    print(f"num_layers = {num_layers}")
    print(f"dropout = {dropout}")
    print(f"bidirectional = {bidirectional}")
    print(f"num_classes = {num_classes}")
    
    # CHECK_PT: recovery model structure complete
    # input("\n=> recovery model structure complete, press Enter to continue")
    
    # DO: add some parameters
    num_batches = 0 # how many batches are generated by dataloader
    
    # DO: create a List[Dict[]] with same shape with test_file
    pred_result = json.loads(test_path.read_text())
    # print(pred_result[90],pred_result[90]["id"],pred_result[90]["text"])
    # CHECK_PT: copy test.json load by function:"json.loads(test_path.read_text())"
    # input("\n=> copy test.json complete, press Enter to continue") 
    
    # TODO: predict dataset
    model.eval()
    with torch.no_grad(): 
        
        t_start = time.time() # save start time for a new epoch
        
        batch_bar = tqdm(dataloaders["test"], desc="Test: ")
        for batch_index, data in enumerate(batch_bar):
            sentence, id = data # sentence.shape = [batch_size, sentence_max_len]       
            sentence = sentence.to(device)
            
            # DO: predict, calculate loss of this batch, and accumulate the loss
            pred = model(sentence) # return softmax(IntentCls_RNN) or logsoftmax(IntentCls_LSTM) for each class
                                   # pred.shape = [batch_size, intent_classes]
                
            # DO: calculate accuracy of this batch, and accumulate the accuracy
            total = id.shape[0]
            for i in range(len(pred)): # calculate match item
                _, pred_cls = torch.max(pred[i],0) # return ["highest probability value" in pred], [index of "highest probability value" in pred]==predicted_intent_class
                pred_in_int = int(pred_cls.detach().cpu()) # turn predicted result to cpu to int
                pre_in_str = test_set.index_to_label(pred_in_int)
                # CHECK_PT: Get Predicted result in str
                # input("\n=> Get Predicted result in str")
                
                # DO: write predict intent back to test file, and wirte to "csv_out" file in specific format mentioned in HW1
                tqdm.write(f"test: before combine, text+id = {pred_result[id[i]]}, intent = {pre_in_str}")
                pred_result[id[i]]["intent"] = pre_in_str
                tqdm.write(f"test: after combine,, result = {pred_result[id[i]]}")
                
                # CHECK_PT: one sentence process complete
                # input(f"\n=> one sentence process complete, data_index = {i}, press Enter to continue")
                
            # CHECK_PT: one batch process complete
            # input(f"\n=> one batch process complete, batch_index = {batch_index}, press Enter to continue")
            num_batches += 1
        # end : batch
        
        # DO: update Info to CMD
        tqdm.write("="*100)
        tqdm.write("") # # write empty new_line
        
        t_stop = time.time() # update the stop time for this epoch
        cost_min = (t_stop - t_start) / 60 
        cost_sec = (t_stop - t_start) % 60
        
        # DO: update Info to CMD
        tqdm.write(f"Test: number of batches = {num_batches}, time_cost = {math.floor(cost_min):.0f} m {math.floor(cost_sec):.0f} s")
        tqdm.write("") # write empty new_line
        tqdm.write("="*100)
        
        # CHECK_PT: all batches process complete
        # input(f"\n=> all batches process complete, press Enter to continue")
        
    # DO: write prediction to file (args.pred_file)
    write_csv(csv_out_path=args.pred_file, pred_result=pred_result)
    # CHECK_PT: release GPU cache complete
    # input(f"\n=> write prediction to '{args.pred_file}' complete, press Enter to continue")
    
    # DO: clean GPU cache
    if args.device != "cpu":
        torch.cuda.empty_cache()
        print("torch.cuda.empty_cache()")
    # CHECK_PT: release GPU cache complete
    # input(f"\n=> release GPU cache complete, press Enter to continue")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file. (file_extension = .json)",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to load model_checkpoint. (file_extension = .ckpt)",
        required=True
    )
    parser.add_argument(
        "--pred_file", 
        type=Path, 
        help="Path to save predict intent. (file_extension = .csv)",
        default="./pred_intent.csv"
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:{device_num}, e.g.cuda:0", default="cuda:1"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
