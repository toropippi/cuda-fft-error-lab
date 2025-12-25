# Experiment 1 – 多倍長整数畳み込みと FFT 誤差

## 構成
- `experiment1.cu`: CUDA 実装（1 ファイル完結）
- `data/`: 実行時に生成される CSV（係数列と summary）
- `python/`: 後段の解析スクリプトを置く場所（未実装）
- `figures/`: 解析結果の図を保存する場所

## ビルド & 実行
```bash
cd experiment1
nvcc -O3 -std=c++17 experiment1.cu -o experiment1_run
./experiment1_run --digits 2048,8192 --output data
```

### オプション
- `--digits`: 解析したい桁数（カンマ区切り）
- `--base`: 桁の基数（デフォルト 10）
- `--output`: CSV を出力するディレクトリ（既定 `data`）
- `--seed`: 乱数シード

`data/summary.csv` と `data/digits_<n>_coefficients.csv` が生成され、STDOUT で平均誤差などを確認できます。

## Python 解析
```bash
cd experiment1/python
python analyze_experiment1.py --data ../data --figures ../figures
```
- `analysis_summary.csv`: rounded 誤差率や max digit error などの統計を `experiment1/data/` に出力
- `figures/`: エラーレート棒グラフ、桁差棒グラフ、桁数ごとのヒストグラム／エラー折れ線を保存
- 相対誤差は `exact_int == 0` の列を除外して計算
