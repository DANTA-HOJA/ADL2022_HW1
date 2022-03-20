# wget "one drive share link"+"&download=1"

# download intent_cls model

if [ ! -d "./download_ckpt/intent/"]; then
    mkdir -p ./download_ckpt/intent/ # mkdir + -p 可建立多層級目錄
fi

wget "https://ntucc365-my.sharepoint.com/:u:/g/personal/r10945018_ntu_edu_tw/EQMgep3Tj51OgLafQPhoVCUBynUbXTI08gmQeXFW7X1eKQ?e=IDO6ee&download=1" -O "./download_ckpt/intent/intent_cls_8800.ckpt"
wget "https://ntucc365-my.sharepoint.com/:u:/g/personal/r10945018_ntu_edu_tw/EQqXooqY7jpOt--lq-EF6jkB7sh9ftV-rVdAVUC2zdE8yg?e=zM2mcR&download=1" -O "./download_ckpt/intent/intent_cls_8946.ckpt"

# download slot_tag model
