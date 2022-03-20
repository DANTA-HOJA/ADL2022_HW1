# ADL2022_HW1
110.2 CSIE5431_深度學習之應用（ADL）


# ---Warning--- 
- Change all **"python3"** back to **"python"** in all **.sh** file in folder.


# LSTM Model summary, @batch_size=256 
    h_0.shape = torch.Size([4, 256, 512])
    after embedding_layer, embedding_layer.shape = torch.Size([256, 128, 300])
    after rnn, h_n.shape = torch.Size([4, 256, 512])
    after fc, output.shape = torch.Size([4, 256, 150])
    after select, output.shape = torch.Size([256, 150])
    after softmax, output.shape = torch.Size([256, 150])


# RUN TRAIN

    python3 train_intent.py --batch_size #(default=256) --hidden_size #(default=512)

# RUN TEST
    python3 test_intent.py --test_file ./data/intent/test.json

output_file (default setting)：**[location = "."]**,　**[file_name = "pred_intent.csv"]**


# Colab: Using "glove.840B.300d"
1.　Download：

    !wget http://nlp.stanford.edu/data/glove.840B.300d.zip
2.　Unzip：

    unzip glove.840B.300d.zip


# Result
- origin：

        python3 train_intent.py --batch_size 64 --hidden_size 768
        python3 "test_(plot_training_logger).py" --ckpt_path "./ckpt/intent/LSTM_best_weight_(b_size64)_(h_size768)_(5_epoch_complete).ckpt"
        python3 test_intent.py --test_file ./data/intent/test.json --ckpt_path "./ckpt/intent/LSTM_best_weight_(b_size64)_(h_size768)_(5_epoch_complete).ckpt"

  - model information & performance：***LSTM_best_weight_(b_size64)_(h_size768)_(epoch5).ckpt***
    ```
    - batch_size = **64**
    - hidden_size = 768
    - num_layers = 2
    - dropout = 0.1
    - bidirectional = True
    - num_classes = 150
    
    - best_result @ epoch 5 (5 in 1:100):
    - best_avg_acc = 88.87%
    - best_avg_loss = 0.4735032820955236
    
    - ==> kaggle = 88.000%
    ```
- improve：

        python3 train_intent.py --batch_size 128 --hidden_size 768
        python3 "test_(plot_training_logger).py" --ckpt_path "./ckpt/intent/LSTM_best_weight_(b_size128)_(h_size768)_(11_epoch_complete).ckpt"
        python3 test_intent.py --test_file ./data/intent/test.json --ckpt_path "./ckpt/intent/LSTM_best_weight_(b_size128)_(h_size768)_(11_epoch_complete).ckpt"
  - model information & performance：***LSTM_best_weight_(b_size128)_(h_size768)_(11_epoch_complete).ckpt***
    ```
    - batch_size = **128**
    - hidden_size = 768
    - num_layers = 2
    - dropout = 0.1
    - bidirectional = True
    - num_classes = 150

    - best_result @ epoch 11 (11 in 1:100):
    - best_avg_acc = 90.38%
    - best_avg_loss = 0.4650507140904665
    ```