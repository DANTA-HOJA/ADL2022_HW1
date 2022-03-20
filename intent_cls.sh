echo -e "Test file <= ${1}\n" # -e 配 \n 可以換行
echo -e "Predict save to => ${2}\n"
echo -e "Model = LSTM_best_weight_(h_size768)_(b_size64)_(5_epoch).ckpt @ ./ckpt/intent/"

python3 test_intent.py --test_file ${1} --pred_file ${2} --ckpt_path "./download/intent_cls_8946.ckpt"