# HOW TO TRAIN：train_intent.py

- To train, please prepare the file list below：
    
  - ```glove.840B.300d.txt``` ---> pre-trained **word embedding** file

  - ```train.json```, ```eval.json``` ---> **train** and **eval** dataset

  - ```embeddings.pt```, ```intent2idx.json```, ```vocab.pkl``` ---> file generated after running **preprocess.sh**

- Training file **default** directory：

  - data directory → ```./data/intent/```（2 files dependencies：```train.json```, ```eval.json```）

  - preprocess.sh generated directory and files → ```./cache/intent/```（3 files dependencies：```embeddings.pt```, ```intent2idx.json```, ```vocab.pkl```）

  - model_checkpoint save directory → ```./ckpt/intent/```（train one time will get one ```.ckpt``` file）

- To reproduce **intent_cls**, kaggle_accuracy = **88.000%**：

      python3 train_intent.py --batch_size 64 --hidden_size 768 --num_epoch 30

- To reproduce **intent_cls**（improve_ver.）, kaggle_accuracy = **89.466%**：

      python3 train_intent.py --batch_size 128 --hidden_size 768 --num_epoch 30

- **args** can be adjusted in **train_intent.py** (default values are list below)：
  ```
    # data
    ☆ --max_len = 128

    # model
    ☆ --hidden_size = 512
    ☆ --num_layers = 2
    ☆ --dropout = 0.1
    ☆ --bidirectional = True
    
    # optimizer
    ☆ --lr = 1e-3

    # dataloader
    ☆ --batch_size = 256

    # training
    ☆ --device = cuda:1
    ☆ --num_epoch = 100
  ```