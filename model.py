from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from zmq import device


class IntentCls_RNN(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int, # call IntentClsDataset.num_classes()
        device : torch.device,
    ) -> None:
        super(IntentCls_RNN, self).__init__()
        
        # DO: Defining some parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.device = device
        
        # DO: Set some parameters
        if bidirectional:
            self.num_directions = 2 # same describe in torch doc.
        embeddings_dim = embeddings.shape[1]
        
        # DO: Defining the layers
        self.embedding_layer = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = nn.RNN(input_size=embeddings_dim ,hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes) # Fully connected layer
        self.softmax = nn.Softmax(dim=1) # Softmax layer 

    def forward(self, x) -> Dict[str, torch.Tensor]:
        
        # DO: Initializing hidden state for first input using method defined below
        batch_size = x.size(0)
        h_0 = self.init_hidden(batch_size)
        #print(f"h_0.shape = {h_0.shape}")
        
        # DO: embedding layer
        # To pass to embedding_layer x.type() must be tensor.LongTensor or tensor.cuda.LongTensor
        embedding_layer = self.embedding_layer(x.long())
        #print(f"after embedding_layer, embedding_layer.shape = {embedding_layer.shape}")
        
        # DO: rnn layer
        _, h_n = self.rnn(embedding_layer, h_0)
        #print(f"after rnn, h_n.shape = {h_n.shape}")
        
        # DO: fully connected layer
        output = self.fc(h_n)
        #print(f"after fc, output.shape = {output.shape}")
        # print(output)
        
        # DO: select last layer of total layers
        # total layer = (bidirectional+num_layers), select the last layer
        output = output[-3,:,:] 
        #print(f"after select, output.shape = {output.shape}")
        # print(output)
        
        # DO: softmax
        output = self.softmax(output)
        #print(f"after softmax, output.shape = {output.shape}")
        # print(output)
        
        return output

    def init_hidden(self, batch_size):
        # DO: initialize a pretrain hidden weight to RNN
        # Is not necessary，pytorch will initialized automatically, or you can send your hidden(pretrain_weight) to model
        hidden = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)
        
        return hidden
    
    def MODEL_TYPE(self) -> str: # Retrun "Model Architecture" => Use to name .ckpt file
        return "RNN"

class IntentCls_LSTM(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_classes: int, # call IntentClsDataset.num_classes()
        device : torch.device,
    ) -> None:
        super(IntentCls_LSTM, self).__init__()
        
        # DO: Defining some parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_directions = 1
        self.device = device
        
        # DO: Set some parameters
        if bidirectional:
            self.num_directions = 2 # same describe in torch doc.
        embeddings_dim = embeddings.shape[1]
        
        # DO: Defining the layers
        self.embedding_layer = Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(input_size=embeddings_dim ,hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.num_classes) # Fully connected layer
        self.logsoftmax = nn.LogSoftmax(dim=1) # Softmax layer 
        
    def forward(self, x):
        
        # DO: embedding layer
        # To pass to embedding_layer x.type() must be tensor.LongTensor or tensor.cuda.LongTensor
        embedding_layer = self.embedding_layer(x.long())
        #print(f"after embedding_layer, embedding_layer.shape = {embedding_layer.shape}")
        
        # DO: lstm layer: torch doc => Outputs: output, (h_n, c_n), output containing all "h_t" for each t.
        lstm_out, hidden_n = self.lstm(embedding_layer)
        #print(f"after lstm, lstm_out.shape = {lstm_out.shape}")
        #print(f"after lstm, h_n.shape = {hidden_n[0].shape}, c_n.shape = {hidden_n[1].shape}")
        
        # DO: fully connected layer
        output = self.fc(lstm_out)
        #print(f"after fc, output.shape = {output.shape}")
        #print(output)
        
        # DO: logsoftmax
        output = self.logsoftmax(output)
        #print(f"after logsoftmax, output.shape = {output.shape}")
        #print(output)
        
        # DO: select last layer of total layers
        # total layer = (bidirectional+num_layers), select the last layer
        output = output[:,-1,:]
        #print(f"after select, output.shape = {output.shape}")
        #print(output)
        
        return output
    
    def MODEL_TYPE(self) -> str: # Retrun "Model Architecture" => Use to name .ckpt file
        return "LSTM"