from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from model import IntentCls_RNN, IntentCls_LSTM
import matplotlib.pyplot as plt

def main(args):

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
    input("\n=> device setting complete, press Enter to continue") 

    # DO: recovery model and training info
    # 1. load full stored file = model+training_info
    ckpt_path = args.ckpt_path # .ckpt file path
    training_info = torch.load(ckpt_path) # load .ckpt file
    # 2. load embedding
    embeddings = torch.load("./cache/intent/embeddings.pt")
    embeddings = embeddings.to(device)
    # 3. load "model parameters"
    batch_size = training_info["batch_size"]
    hidden_size = training_info["model_para"]["hidden_size"]
    num_layers = training_info["model_para"]["num_layers"]
    dropout = training_info["model_para"]["dropout"]
    bidirectional = training_info["model_para"]["bidirectional"]
    num_classes = training_info["model_para"]["num_classes"]
    # 4. initialize same model structure store in ckpt file
    # ==> because we only save model gradient (using "model.state_dict()") not whole "model info(include "model structure")"
    model = IntentCls_LSTM(embeddings=embeddings, hidden_size=hidden_size, num_layers=num_layers,
                            dropout= dropout, bidirectional= bidirectional,
                            num_classes=num_classes, device=device) # init model
    model.load_state_dict(training_info['model_state_dict']) # load model 
    model = model.to(device) # send model to device
    print(f"{model}\n")
    print(f"batch_size = {batch_size}")
    print(f"hidden_size = {hidden_size}")
    print(f"num_layers = {num_layers}")
    print(f"dropout = {dropout}")
    print(f"bidirectional = {bidirectional}")
    print(f"num_classes = {num_classes}\n")
    
    # 5. load loggers
    Epoch_loss_logger = {'train': training_info['trian_loss'], 'eval': training_info['eval_loss']}
    Epoch_acc_logger = {'train': training_info['trian_acc'], 'eval': training_info['eval_acc']}

    # DO: show best information
    print(f"best_result @ epoch {training_info['epoch'][0]} ({training_info['epoch'][0]} in 1:{training_info['epoch'][1]}):")
    print(f"best_avg_acc = {training_info['best_avg_acc']:.2%}")
    print(f"best_avg_loss = {training_info['best_avg_loss']}")

   # DO: draw and save graph
    # 1. loss graph
    plt.figure("Epoch_loss")
    plt.title("Epoch_loss")
    plt.plot(Epoch_loss_logger['train'], label="train")
    plt.plot(Epoch_loss_logger['eval'], label="eval")
    plt.legend()
    plt.savefig("epoch_Loss_logger.png")
    # 2. accuracy graph
    plt.figure("Epoch_acc")
    plt.title("Epoch_acc")
    plt.plot(Epoch_acc_logger['train'], label="train")
    plt.plot(Epoch_acc_logger['eval'], label="eval")
    plt.legend()
    plt.savefig("epoch_Acc_logger.png")
    
    # DO: clean GPU cache
    if args.device != "cpu":
        torch.cuda.empty_cache()
        print("torch.cuda.empty_cache()")
    # CHECK_PT: release GPU cache complete
    input(f"\n=> release GPU cache complete, press Enter to continue")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint. (file_extension = .ckpt)",
        required=True
    )
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:{device_num}, e.g.cuda:0", default="cuda:1"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)






