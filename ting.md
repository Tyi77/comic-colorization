# Style2Paints V4.5

## Install

適合版本：CUDA 10.0, CuDNN 7, Python 3.6

``` shell
    pip install opencv-contrib-python
    pip install tensorflow
    pip install bottle
    pip install tqdm
```

從 [雲端](https://drive.google.com/drive/folders/1mM8RbNYM0AHu31U6UaEroGFT1hOHlY0V?usp=sharing) 下載 model 並放進 s2p_v45_server/nets 資料夾中，如：

``` shell
    s2p_v45_server/nets/inception.net
    s2p_v45_server/nets/gau.npy
    s2p_v45_server/nets/refs.net
    ...
```

## Run

``` shell
    cd s2p_v45_server
    python Style2PaintsV45_source.py
```

# inkn'hue

## Setup
1. 進入到資料夾

``` shell
cd inknhue-main
```

2. 設定環境
適合版本：Python 3.10、cudatoolkit >= 11.8

``` shell
conda env create -f environment.yaml
conda activate inknhue
```


3. 從 Hugging Face 下載 models

``` shell
rm -r models
git lfs install
git clone https://huggingface.co/rossiyareich/inknhue models
```

## Run
``` shell
python app.py
```