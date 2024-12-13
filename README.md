# Comic-Colorization

## Evaluation Metric

在我們的研究中，為了全面評估模型的表現，我們選擇了 Fréchet Inception Distance (FID) 和峰值信噪比 (PSNR) 作為主要的量化評估指標。

### FID 指標的介紹
FID 用於衡量生成圖像的分佈與真實圖像的分佈之間的相似性，是一種基於 Wasserstein 距離的評估方法。FID 利用預訓練的 InceptionV3 模型，通過提取中間層激活值來建構生成圖像與真實圖像的多維高斯分佈，並計算兩者之間的距離。

### FID 的計算細節
- **數據分佈建構**：我們使用測試數據估算生成圖像與真實圖像的高斯分佈參數（均值和標準差矩陣）。
- **距離計算**：FID 的數學公式如下：
  ```
  FID = ||μ_r - μ_g||^2 + Tr(Σ_r + Σ_g - 2(Σ_rΣ_g)^{1/2})
  ```
  其中：  
  - μ_r 和 μ_g 分別為真實圖像和生成圖像分佈的均值。
  - Σ_r 和 Σ_g 分別為真實圖像和生成圖像分佈的協方差矩陣。

### PSNR 指標的介紹
PSNR 是一種用於衡量圖像質量的常見指標，特別是在壓縮和生成影像領域。PSNR 通常通過計算峰值信噪比來評估生成圖像與真實圖像之間的相似性。PSNR 的單位為分貝 (dB)，值越高代表生成圖像與真實圖像的相似度越高。

### 結論
綜合考量後，我們決定採用 FID跟PSNR 作為主要評估指標，通過該指標量化我們方法的改進效果，並輔助分析生成模型的性能。

### 參考資料
[https://wandb.ai/authors/One-Shot-3D-Photography/reports/-Frechet-Inception-Distance-FID-GANs---Vmlldzo0MzI0MjA](https://wandb.ai/authors/One-Shot-3D-Photography/reports/-Frechet-Inception-Distance-FID-GANs---Vmlldzo0MzI0MjA)

## How to Use
1. 把生成的圖片放在 `output` 資料夾。
2. 把真實圖片 (ground truth) 放在 `real` 資料夾。
3. 只要直接執行FID.py或PSNR.py就可以得到兩個分數了

