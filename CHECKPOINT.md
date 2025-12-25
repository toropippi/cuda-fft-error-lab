## Project Checkpoints

### Context
- Goal: build CUDA/C++ programs plus Python analysis to evaluate float vs CUDA fast math trig for FFT + rotations via four experiments (multi-precision integer convolution, FFT accuracy, rotational stability, Monte Carlo norm error).
- Target GPU: RTX 5090 (CC >= 8.x), toolchain: `nvcc`.
- Precision conditions: reference double FFT, float FFT using precomputed twiddles (high precision), float FFT using `__sinf/__cosf` (fast).
- Outputs: CSV/binary data that Python ingests to plot figures (PNG) and compute summary tables written as CSV.

### Completed
- Parsed `codex_prompt.txt` and captured core requirements.
- Reorganizedレポジトリ: 実験ごとに専用ディレクトリを作成（`experiment1/` 配下に CUDA コード・データ・解析枠を集約）。
- Experiment 1 専用単一ファイル CUDA 実装 (`experiment1/experiment1.cu`) を整備し、`experiment1/data` へ係数 CSV と summary を出力。
- Experiment 1 向け Python 解析スクリプト (`experiment1/python/analyze_experiment1.py`) を追加し、エラーレート棒グラフ・ヒストグラム・index 対誤差を `figures/` に保存できるようにした。

### Project Structure Plan
- `experiment1/`: CUDA実装 + data/ + python/ + figures/。
- `experiment2/` 以降も同様にフォルダを増やし、それぞれ独立した `.cu` と Python 解析を配置予定。
- `CHECKPOINT.md`: 全体の進捗と残タスクを記録。

### Next Steps
1. 実験 2〜4 用のディレクトリを順次追加し、各 `.cu` と Python 解析を独立構成で実装する。
2. 全体のビルド・実行・解析手順を整理し、チェックポイントを適宜更新する。
