# ADL2022_HW1
110.2 CSIE5431_深度學習之應用（ADL）

Change "python3" back to "python" in preprocess.sh file.


# Model summary, @batch_size=256 
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

output_file (default setting) : ___[location = "."],　[file_name="pred_intent.csv"]___


# Colab: Using "glove.840B.300d"
1.　Download：

    !wget http://nlp.stanford.edu/data/glove.840B.300d.zip
2.　Unzip：

    unzip glove.840B.300d.zip


# TESTING
    python3 train_intent.py --batch_size 64 --hidden_size 768
    python3 "test_(plot_training_logger).py" --ckpt_path "./ckpt/intent/LSTM_best_weight_(h_size768)_(b_size64)_(5_epoch).ckpt"
    python3 test_intent.py --test_file ./data/intent/test.json --ckpt_path "./ckpt/intent/LSTM_best_weight_(h_size768)_(b_size64)_(5_epoch).ckpt"