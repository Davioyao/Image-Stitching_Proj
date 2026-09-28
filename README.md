# Image-Stitching_Proj

原地旋轉拍攝的多張照片 → 全景拼接 ＋ 色彩/曝光平衡 ＋ 拍照位置逆推（雷達俯視圖），
並以可互動的 Tkinter GUI 呈現。

<img src="Result/Image-stitching-app.png" width="800">

## 拼接成果

![Stitcher 全景輸出](Result/Stitcher_Full.jpeg)

`Result/` 另有手刻鏈全景（`Full_View.jpeg`）與雷達俯視圖存檔（`Bird-view.png`）。

## 環境安裝

```bash
conda activate opencv_evk
pip install -r requirements.txt
```

需求：`opencv-contrib-python>=4.8.0`、`numpy>=1.24.0`、`Pillow>=10.0.0`

## 使用方式

### GUI 模式

```bash
python main.py                    # 預設讀取 ./Pic
python main.py --folder ./Pic --width 1240
```

> `Pic/` 原始照片不上傳 GitHub，請自備有 5%~30% 重疊的連拍照片放入 `Pic/` 再執行。

### CLI 模式

```bash
python main.py --cli --save pano.jpg
python main.py --cli --method stitcher --save pano_st.jpg
```

| 參數 | 說明 |
|---|---|
| `--width` | 工作寬度 px（預設 1240）：先等比縮圖再拼接，越大越細但越慢 |
| `--method` | `auto`（手刻優先、失敗轉 Stitcher）／`manual`／`stitcher` |
| `--no-balance` | 關閉色彩平衡，改簡單平均（接縫對照用） |
