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

    # DO: load embedding
    embeddings = torch.load("./cache/intent/embeddings.pt")
    embeddings = embeddings.to(device)

    # initialize same model structure store in ckpt file
    # because we only save model parameters (using "model.state_dict()") not whole "model info(include "model structure")"
    model = IntentCls_RNN(embeddings=embeddings, hidden_size=512, num_layers=2,
                            dropout= 0.1, bidirectional= True,
                            num_classes=150, device=device) # init model
    model = model.to(device) # send model to device
    print(model)

    # DO: recovery model and logger
    model_path = f"./ckpt/intent/{model.MODEL_TYPE()}_best_weight_(100_epoch_complete).ckpt" # .ckpt file path
    checkpoint = torch.load(model_path) # load .ckpt file

    # load model parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    Epoch_loss_logger = {'train': checkpoint['trian_loss'], 'eval': checkpoint['eval_loss']}
    Epoch_acc_logger = {'train': checkpoint['trian_acc'], 'eval': checkpoint['eval_acc']}

    print(f"best_result @ epoch {checkpoint['epoch'][0]} ({checkpoint['epoch'][0]} in 1:{checkpoint['epoch'][1]}):")
    print(f"best_avg_acc = {checkpoint['best_avg_acc']:.2%}")
    print(f"best_avg_loss = {checkpoint['best_avg_loss']}")

    plt.figure("Epoch_loss")
    plt.plot(Epoch_loss_logger['train'])
    plt.plot(Epoch_loss_logger['eval'])
    plt.legend(['train', 'eval'])
    plt.title('loss')
    plt.savefig("epoch_Loss_logger.png")

    plt.figure("Epoch_acc")
    plt.plot(Epoch_acc_logger['train'])
    plt.plot(Epoch_acc_logger['eval'])
    plt.legend(['train', 'eval'])
    plt.title('accuracy')
    plt.savefig("epoch_Acc_logger.png")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to the test file.",
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






