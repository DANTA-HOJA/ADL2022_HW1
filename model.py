from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding
from torch.nn import functional as F
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
        
        # Defining some parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.device = device
        
        # Set some parameters
        if bidirectional:
            self.num_directions = 2 # same describe in torch doc.
        embeddings_dim = embeddings.shape[1]
        
        #Defining the layers
        self.embedding_layer = Embedding.from_pretrained(embeddings, freeze=False)
        self.rnn = nn.RNN(input_size=embeddings_dim ,hidden_size=hidden_size, num_layers=num_layers,
                          dropout=dropout, bidirectional=bidirectional, batch_first=True)
        
        # Fully connected and Softmax layer 
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x) -> Dict[str, torch.Tensor]:
        
        # Initializing hidden state for first input using method defined below
        batch_size = x.size(0)
        h_0 = self.init_hidden(batch_size)
        #print(f"h_0.shape = {h_0.shape}")
        
        # embedding layer
        # pass to embedding_layer x.type() must be tensor.LongTensor or tensor.cuda.LongTensor
        embedding_layer = self.embedding_layer(x.long())
        #print(f"after embedding_layer, embedding_layer.shape = {embedding_layer.shape}")
        
        # rnn layer
        _, h_n = self.rnn(embedding_layer, h_0)
        #print(f"after rnn, h_n.shape = {h_n.shape}")
        
        # fully connected layer
        output = self.fc(h_n)
        #print(f"after fc, output.shape = {output.shape}")
        #print(output)
        
        output = output[-1,:,:] # total layer = 4(bidirectional+num_layers), choose the last layer
        # print(f"after select, output.shape = {output.shape}")
        # print(output)
        
        # softmax
        output = self.softmax(output)
        # print(f"after softmax, output.shape = {output.shape}")
        # print(output)
        
        return output

    def init_hidden(self, batch_size):
        # To send a pretrain hidden weight to RNN
        # Is not necessary，pytorch will initialized automatically
        hidden = torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size).to(self.device)
        
        return hidden
    

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
        
        # Defining some parameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_directions = 1
        self.device = device
        
        # Set some parameters
        if bidirectional:
            self.num_directions = 2 # same describe in torch doc.
        embeddings_dim = embeddings.shape[1]
        
        # Defining the layers
        self.embedding_layer = Embedding.from_pretrained(embeddings, freeze=False)
        self.lstm = nn.LSTM(input_size=embeddings_dim ,hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # Fully connected and Softmax layer 
        self.fc = nn.Linear(self.num_directions*self.hidden_size, self.num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        
        # embedding layer
        # pass to embedding_layer x.type() must be tensor.LongTensor or tensor.cuda.LongTensor
        embedding_layer = self.embedding_layer(x.long())
        #print(f"after embedding_layer, embedding_layer.shape = {embedding_layer.shape}")
        
        # LSTM layer: torch doc => Outputs: output, (h_n, c_n), output containing all "h_t" for each t.
        lstm_out, hidden_n = self.lstm(embedding_layer)
        #print(f"after lstm, lstm_out.shape = {lstm_out.shape}")
        #print(f"after lstm, h_n.shape = {hidden_n[0].shape}, c_n.shape = {hidden_n[1].shape}")
        
        # fully connected layer
        output = self.fc(lstm_out)
        #print(f"after fc, output.shape = {output.shape}")
        #print(output)
        
        # logsoftmax
        output = self.logsoftmax(output)
        #print(f"after logsoftmax, output.shape = {output.shape}")
        #print(output)
        
        output = output[:,-1,:] # total layer = 4(bidirectional+num_layers), choose the last layer
        #print(f"after select, output.shape = {output.shape}")
        #print(output)
        
        return output