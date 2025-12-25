# Experiment 2 – 単一 FFT 誤差解析

## 概要
- ランダム、単一正弦波、正弦波の和という 3 種類の信号に対して FFT を 1 回実行し、CPU double FFT を正解として GPU の 2 条件（LUT／高速近似）を比較します。
- 各ビンの複素スペクトルと絶対／相対誤差を CSV に保存し、GPU 逆 FFT の再構成誤差（L2 ノルム）も計測します。

## ディレクトリ構成
- `experiment2.cu`: CUDA 実装（単体実行で完結）
- `data/`: `signal_<type>_n<length>_spectrum.csv` と `experiment2_summary.csv` を出力
- `python/`: 後段解析用（未実装）
- `figures/`: Python 解析結果の図を保存する場所

## ビルド & 実行
```bash
cd experiment2
nvcc -O3 -std=c++17 experiment2.cu -o experiment2_run
./experiment2_run --lengths 1024,65536 --signals random,sine,sine_sum --output data
```

### 主なオプション
- `--lengths`: FFT サイズ（2 のべき乗のみ、カンマ区切り）。既定 `1024,65536`
- `--signals`: 入力信号の種類（`random`, `sine`, `sine_sum`）。既定 3 種
- `--output`: CSV 出力先ディレクトリ（既定 `data`）
- `--seed`: 乱数シードのベース値（既定 `2024`）

## 出力
- `signal_<type>_n<length>_spectrum.csv`: 参照スペクトルと LUT/fast の複素値、絶対誤差・相対誤差を記録
- `experiment2_summary.csv`: まとめ統計（平均／最大の絶対・相対誤差、逆 FFT の L2 誤差）を出力
- 実行中に各ケースの統計を STDOUT に表示するので、その場で誤差感度を把握できます。
