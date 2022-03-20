echo -e "Test file <= ${1}\n" # -e 配 \n 可以換行
echo -e "Predict save to => ${2}\n"
echo -e "Model = intent_cls_8946.ckpt @ ./download/"

python3 test_intent.py --test_file ${1} --pred_file ${2} --ckpt_path "./download_ckpt/intent/intent_cls_8946.ckpt"