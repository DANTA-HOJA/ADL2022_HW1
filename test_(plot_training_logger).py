import torch
import torch.optim as optim
from torch import nn

from model import IntentCls_RNN, IntentCls_LSTM

import matplotlib.pyplot as plt


# DO: check if CUDA is available
if not torch.cuda.is_available():
    print("QQ, you should buy a graphic card ~, device chang to 'cpu'\n")
    device = "cpu"
else:
    device = "cuda:0"
device = torch.device(device)
print(f'==> device using {device}')

# DO: load embedding
embeddings = torch.load("./cache/intent/embeddings.pt")
embeddings = embeddings.to(device)

# initialize same model structure store in ckpt file
# because we only save model parameters (using "model.state_dict()") not whole "model info(include "model structure")"
model = IntentCls_RNN(embeddings=embeddings, hidden_size=512, num_layers=2,
                        dropout= 0.1, bidirectional= True,
                        num_classes=150, device=device) # init model

# DO: recovery model and logger
model_path = f"./ckpt/intent/{model.MODEL_TYPE()}_best_weight_(100_epoch_compelet).ckpt" # .ckpt file path
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







