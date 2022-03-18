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



# RUN TEST
    python3 test_intent.py --test_file ./data/intent/test.json

> output_file (default setting) : location = ".", file_name="pred_intent.csv"