# nanochat

![nanochat logo](../dev/nanochat.png)
![scaling laws](../dev/scaling_laws_jan26.png)

nanochat は、LLM を訓練するための最もシンプルな実験用ハーネスです。**単一の GPU ノード**上で動作するように設計されており、コードは最小限でハックしやすく、トークナイゼーション・事前学習・ファインチューニング・評価・推論・チャット UI まで、LLM の主要なステージをすべてカバーしています。

たとえば、2019 年に約 43,000 ドルかかった GPT-2 相当の能力を持つ LLM を、わずか **48 ドル**（8×H100 GPU ノードで約 2 時間）で訓練し、ChatGPT 風の Web UI で会話することができます。スポットインスタンスを使えば、コストは約 15 ドルまで下がります。

より一般的には、nanochat は `--depth`（GPT Transformer モデルのレイヤー数）という**単一の複雑度ダイヤル**を設定するだけで、計算最適なモデルのミニシリーズ全体を訓練できるように構成されています。GPT-2 相当の能力は depth 26 前後に相当します。その他すべてのハイパーパラメータ（Transformer の幅、ヘッド数、学習率調整、訓練ホライズン、重み減衰など）は自動的に最適な値で計算されます。

リポジトリに関する質問は、[DeepWiki](https://deepwiki.com/karpathy/nanochat)、[Discussions タブ](https://github.com/karpathy/nanochat/discussions)、または Discord の [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) チャンネルをご利用ください。

## Time-to-GPT-2 リーダーボード

現在の開発の主な焦点は、最も多くの計算リソースを消費する事前学習ステージのチューニングです。modded-nanogpt リポジトリに触発され、進歩とコミュニティのコラボレーションを促進するために、nanochat は「GPT-2 スピードラン」のリーダーボードを維持しています。これは DCLM CORE スコアで測定される GPT-2 相当の能力に到達するまでの壁時計時間です。[runs/speedrun.sh](../runs/speedrun.sh) スクリプトが常に GPT-2 相当モデルの訓練とチャットのリファレンス手順を反映しています。

| # | 時間 | val_bpb | CORE | 説明 | 日付 | コミット | 貢献者 |
|---|------|---------|------|------|------|----------|--------|
| 0 | 168 時間 | - | 0.2565 | OpenAI GPT-2 オリジナルチェックポイント | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 ベースライン、やや過学習 | 2026/1/29 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 やや訓練不足 **+fp8** | 2026/2/2 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | 総バッチサイズを 1M トークンに拡大 | 2026/2/5 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | データセットを NVIDIA ClimbMix に変更 | 2026/3/4 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch [ラウンド 1](https://x.com/karpathy/status/2031135152349524125) | 2026/3/9 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch ラウンド 2 | 2026/3/14 | a825e63 | @karpathy |

主要な指標は「Time to GPT-2」— 8×H100 GPU ノードで GPT-2 (1.6B) の CORE 指標を超えるのに必要な壁時計時間です。GPT-2 の CORE スコアは 0.256525 です。2019 年に GPT-2 の訓練には約 43,000 ドルかかりましたが、7 年間にわたるスタック全体の多くの進歩により、現在は 100 ドル未満ではるかに高速に実現できるようになりました（例: 現在の約 3 ドル/GPU/hr で、8×H100 ノードは約 24 ドル/hr、2 時間で約 48 ドル）。

詳細は [dev/LEADERBOARD.md](../dev/LEADERBOARD.md) を参照してください。

## セットアップ

nanochat は依存関係管理に [uv](https://docs.astral.sh/uv/) を使用しています。

```bash
uv sync --extra gpu    # CUDA (A100/H100 等) の場合
uv sync --extra cpu    # CPU のみ / MPS の場合
source .venv/bin/activate
```

開発用（pytest, matplotlib, ipykernel, transformers 等を追加）:

```bash
uv sync --extra gpu --group dev
```

## GPT-2 を再現して会話する

最も楽しいのは、自分の GPT-2 を訓練して会話することです。そのためのパイプライン全体は [runs/speedrun.sh](../runs/speedrun.sh) に含まれており、8×H100 GPU ノードで実行するように設計されています。お好みのプロバイダー（例: [Lambda](https://lambda.ai/service/gpu-cloud)）から 8×H100 GPU ボックスを起動し、訓練スクリプトを実行します:

```bash
bash runs/speedrun.sh
```

完了までに約 3 時間かかるため、screen セッション内で実行することをお勧めします。完了後、ChatGPT 風の Web UI で会話できます。ローカルの uv 仮想環境がアクティブであることを確認し（`source .venv/bin/activate`）、以下を実行します:

```bash
python -m scripts.chat_web
```

表示された URL にアクセスしてください。例えば Lambda ではノードのパブリック IP に続けてポート番号を指定します（例: `http://209.20.xxx.xxx:8000/`）。あとは通常の ChatGPT と同じように会話できます！

補足:

- Ampere 8×A100 GPU ノードでも問題なく動作しますが、やや遅くなります。
- `torchrun` を省略して単一 GPU でも動作し、ほぼ同一の結果が得られます（自動的に勾配累積に切り替わります）が、8 倍の時間がかかります。
- GPU の VRAM が 80GB 未満の場合は、`--device-batch-size` を調整する必要があります（例: 32 → 16, 8, 4, 2, 1）。

## 研究

研究者でプレトレーニング改善に興味がある場合は、[runs/scaling_laws.sh](../runs/scaling_laws.sh) と [runs/miniseries.sh](../runs/miniseries.sh) が参考になります。素早い実験（約 5 分のプレトレーニング）には、12 レイヤーモデル（GPT-1 サイズ）の訓練がお勧めです:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

nanochat は `--depth` という単一の複雑度ダイヤルを中心に設計されており、この整数値から他のすべてのハイパーパラメータ（幅、ヘッド数、学習率調整、訓練ホライズン、重み減衰など）が自動的に計算最適な値で決定されます。

## CPU / MPS での実行

[runs/runcpu.sh](../runs/runcpu.sh) に CPU や Apple Silicon での実行例があります。訓練時間を合理的な範囲（数十分）に収めるため、LLM のサイズが大幅に縮小されます。この方法では強い結果は得られません。

## 精度 / dtype

nanochat は `torch.amp.autocast` を使用しません。代わりに、`nanochat/common.py` で定義されるグローバルな `COMPUTE_DTYPE` を通じて精度を明示的に管理します。デフォルトはハードウェアに基づいて自動検出されます:

| ハードウェア | デフォルト dtype | 理由 |
|-------------|-----------------|------|
| CUDA SM 80+（A100, H100 等） | `bfloat16` | ネイティブ bf16 テンソルコア |
| CUDA SM < 80（V100, T4 等） | `float32` | bf16 非対応; `NANOCHAT_DTYPE=float16` で fp16 使用可（GradScaler 使用） |
| CPU / MPS | `float32` | 低精度テンソルコアなし |

環境変数 `NANOCHAT_DTYPE` でデフォルトを上書きできます:

```bash
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"   # fp32 を強制
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train  # bf16 を強制
```

仕組み: モデルの重みは fp32 で保存されますが（オプティマイザの精度のため）、カスタム `Linear` レイヤーがフォワードパス中に `COMPUTE_DTYPE` にキャストします。エンベディングはメモリ節約のため `COMPUTE_DTYPE` で直接保存されます。これにより autocast と同じ混合精度の利点を得つつ、どの精度でどの計算が実行されるかを完全に明示的に制御できます。

注意: `float16` 訓練では `base_train.py` で `GradScaler` が自動的に有効になり、勾配のアンダーフローを防ぎます。SFT もこれをサポートしていますが、RL は現在未対応です。fp16 での推論はどこでも問題なく動作します。

## ライセンス

MIT
