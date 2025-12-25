# CUDA FFT 誤差解析リポジトリ

2 つの CUDA 実験（Experiment 1/2）をまとめたリポジトリです。どちらも自前実装の FFT を用いて GPU 上での誤差特性を記録し、CPU の倍精度 FFT を正解として比較することで、LUT 方式と高速近似方式（`__sincosf`）の差分を可視化します。

## ディレクトリ構成
- `experiment1/`: 多倍長整数の畳み込みを FFT で実行し、係数ごとの差分を CSV/TXT 出力
- `experiment2/`: 単一の FFT（複素スペクトル）の誤差を信号種別ごとに解析
- `figures/`, `data/`: 各実験ディレクトリ配下に生成される解析結果

## ひと目でわかるポイント
1. Experiment 1 は桁数 2,048/8,192 の乱数多倍長整数どうしを畳み込み、各係数の絶対誤差・相対誤差と精度統計を保存します。GPU は float2、CPU は double を使用し、LUT 版のほうが量子化誤差を抑えられる様子が `data/digits_*` で確認できます。
2. Experiment 2 は 3 種類の信号（ランダム、単一サイン、サイン和）×複数の FFT サイズを対象に、1 回の FFT で生じるビン誤差・逆 FFT の再構成誤差 (L2) を比較します。
3. どちらの実験も実行時に CSV/TXT を `data/` へ書き出し、後段の Python スクリプト（`python/` ディレクトリ）で可視化することを想定しています。

## ビルド & 実行
```bash
# Experiment 1
cd experiment1
nvcc -O3 -std=c++17 experiment1.cu -o experiment1_run
./experiment1_run --digits 2048,8192 --output data

# Experiment 2
cd experiment2
nvcc -O3 -std=c++17 experiment2.cu -o experiment2_run
./experiment2_run --lengths 1024,65536 --signals random,sine,sine_sum --output data
```
- `--digits`, `--lengths`, `--signals`, `--base`, `--seed`, `--output` などのオプションは各ディレクトリ内の README を参照してください。
- 解析用 Python スクリプトは `experiment*/python/` に配置し、`--data` / `--figures` オプションで出力先を指定する想定です。

## 公開時のメモ
- 実験バイナリ（`experiment*_run.*`）はビルド済みサンプルです。不要な場合は `.gitignore` などで除外してから公開してください。
- 生成物の例（CSV/図）は `experiment*/data/` や `experiment*/figures/` にあります。必要に応じてサンプルのみ残すか、空ディレクトリにしておくとクリーンです。

## ライセンス・引用
公開リポジトリにする際は、使用しているデータやコードに応じてライセンス表記を追加してください（例: MIT License）。記事やブログに引用する場合は、各実験の README へのリンクを併記すると構成が伝わりやすくなります。
