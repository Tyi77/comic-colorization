# Style2Paints V4.5

用於將我們生成的 Color hint image 繪製到無上色的線稿圖上。

## Setup

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
    s2p_v45_server/nets/refs.net
    ...
```

## Run

``` shell
    cd s2p_v45_server
    python Style2PaintsV45_source.py
```

## How to use

為了讓他可以自動輸入多張圖片並輸出，本來想要在 UI 上多設置一個可以執行這項任務的按鈕，但他的 UI 撰寫方式過於複雜，且其中還牽涉到了擁有千行且完全沒有整理過的 project.js，要更改實在過於困難，於是我們用了別的方式觸發這項任務。觸發方式如下：

1. 點擊右下角的上傳圖片按鈕
2. 隨意放入圖片檔
3. 點擊 OK 鍵觸發

輸入與輸出檔案位址：
可從 Style2PaintsV45_source.py 更改

輸入：
Input Image（可為黑白或彩色）

```shell
folder_path = './append/Dataset'
```

Color Hint Image（要與對應的圖片檔名相同，且為 png 檔）

```shell
color_hint_image_points = extract_color_blocks('./append/ColorHint/color_hint' + file_name + '.png')
```

輸出：
Rough Color Image（我們所要的上色結果）

```shell
cv2_imwrite('./append/Result/' + file_name + '.jpg', blended_smoothed_careless)
```

Sketch Image（因 inkn'hue 對於圖片大小有強烈限制，因此可用於 inkn'hue）

```shell
cv2_imwrite('./append/Sketch/' + file_name + '.jpg', sketch)
```

# inkn'hue

用於解決 Style2Paints 生成的 Rough Color Image 顏色過於飽和及部分溢色問題。

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

## How to use

為了讓他可以自動輸入多張圖片並輸出，我們在原本的第一個頁面的 UI 下方新增了 Run Dataset，直接點擊即可。但由於原本的部分函式回傳資料類型限制，難以實作這項功能，所以更改了回傳的資料，這也導致原功能無法正常運作，內容如下：

皆位於 app.py 中，將回傳值改為註解掉的內容即可恢復正常功能。

```shell
def generate(sketch: Image.Image, s2p: Image.Image):
    ...
    # return gr.Image(result, interactive=True)
    return result
```

```shell
def postprocess(gen: Image.Image, s2p: Image.Image, cratio):
    ...
    # return [ret, ret]
    return combined
```

輸入與輸出檔案位址：
可從 app.py 更改

輸入：
Sketch Image（Style2Paints 生成結果）

```shell
folder1 = './append/Sketch'
```

Rough Color Image（Style2Paints 生成結果）

```shell
folder2 = './append/Style2Paints'
```

輸出：
Color Image（我們所要的上色結果）

```shell
exp_path = os.path.join('./append/Result', file1)
```

可調參數：
CIELAB Interpolation 所需的參數 λa∗b∗，用於調整 inkn'hue 通過 VAE 生成的圖片與 Style2Paints 生成的圖片的插值比例，數值範圍 0 ~ 1，越接近 0 越接近 VAE 生成的圖片；反之，越接近 1 越接近 Style2Paints 生成的圖片。
我們設定數值為 0.8。

```shell
exp_img = postprocess(gen, img2, 0.8)
```
