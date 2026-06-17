# nanochat フォルダ構成と役割まとめ

---

## 1. `nanochat/` — コアライブラリ

モデル定義・学習・推論に必要な全ての基盤コードが格納されている。

### モデル定義

| ファイル | 役割 |
|---|---|
| `gpt.py` | GPTモデル本体。`GPTConfig`, `GPT`, `CausalSelfAttention`, `MLP`, `Block` を定義。Rotary Embeddings, GQA (Group-Query Attention), Sliding Window Attention, Value Embeddings (ResFormer) などを実装 |
| `flash_attention.py` | Flash Attention 3 (Hopper GPU) と PyTorch SDPA の統一インターフェース。`flash_attn` モジュールとして `gpt.py` から利用される |
| `fp8.py` | FP8学習サポート。`Float8Linear` と `convert_to_float8_training` を提供。torchao の約2000行を~150行で代替実装 |

### 学習基盤

| ファイル | 役割 |
|---|---|
| `optim.py` | カスタムオプティマイザ `MuonAdamW` / `DistMuonAdamW`。行列パラメータに Muon (Newton-Schulz直交化)、埋め込み・スカラーに AdamW を使い分ける |
| `dataloader.py` | 分散学習用データローダー。BOS-aligned best-fit packing アルゴリズムで100%トークン利用率を実現 |
| `dataset.py` | 事前学習データセット (ClimbMix-400B) の parquet ファイルダウンロード・イテレーション |
| `checkpoint_manager.py` | チェックポイントの保存・読み込み。`base` / `sft` / `rl` の3フェーズに対応 |

### 推論・評価

| ファイル | 役割 |
|---|---|
| `engine.py` | 効率的な推論エンジン。`KVCache` (Flash Attention 3対応) と `Engine` クラスを実装。バッチ生成・Python計算機ツール使用のステートマシンを内包 |
| `tokenizer.py` | BPEトークナイザー。`HuggingFaceTokenizer` と `RustBPETokenizer` の2実装。会話を token id 列にレンダリングする `render_conversation` / `render_for_completion` を提供 |
| `core_eval.py` | CORE ベンチマーク評価ロジック。multiple choice / schema / language modeling の3タスクタイプに対応 |
| `loss_eval.py` | Bits per byte (bpb) 評価。語彙サイズに依存しない損失指標を計算 |
| `execution.py` | LLM生成コードのサンドボックス実行 (HumanEval評価用)。別プロセス・タイムアウト・メモリ制限付き |

### その他

| ファイル | 役割 |
|---|---|
| `common.py` | 共通ユーティリティ。デバイス自動検出、DDP初期化・クリーンアップ、`COMPUTE_DTYPE` 自動判定、GPU FLOPs テーブル |
| `report.py` | 学習レポート生成。各フェーズの結果を Markdown ファイルに記録し、最終的に `report.md` に集約 |
| `ui.html` | Web チャット用フロントエンド HTML (FastAPI サーバーから配信) |

---

## 2. `scripts/` — 実行スクリプト

学習・評価・推論の各フェーズを実行するエントリーポイント群。`nanochat/` のコードを組み合わせて実際の処理を行う。

### 学習フロー (3段階)

```
tok_train.py → base_train.py → chat_sft.py → chat_rl.py
  トークナイザー  事前学習        SFT           強化学習
```

### スクリプト一覧

| ファイル | 役割 |
|---|---|
| `tok_train.py` | BPEトークナイザーの学習。ClimbMix データから最大2B文字を使い vocab_size=32768 のトークナイザーを訓練。`token_bytes.pt` も生成 |
| `tok_eval.py` | トークナイザーの圧縮率評価。GPT-2 / GPT-4 との比較表を出力 |
| `base_train.py` | ベースモデルの事前学習。スケーリング則に基づくバッチサイズ・学習率・weight decay の自動計算、FP8学習、DDP対応 |
| `base_eval.py` | ベースモデルの評価。CORE metric / BPB / サンプリングの3モード。HuggingFace モデルの評価にも対応 |
| `chat_sft.py` | Supervised Fine-Tuning。SmolTalk・MMLU・GSM8K・SpellingBee などのデータ混合で会話能力を付与 |
| `chat_rl.py` | 強化学習 (GRPO/REINFORCE)。GSM8K の正解/不正解を報酬として、数学的推論能力を強化 |
| `chat_eval.py` | チャットモデルの評価。ARC / MMLU / GSM8K / HumanEval / SpellingBee を実行し ChatCORE スコアを算出 |
| `chat_cli.py` | コマンドラインチャットインターフェース。対話モードまたは `--prompt` による1回応答 |
| `chat_web.py` | FastAPI ベースの Web チャットサーバー。マルチ GPU ワーカープールで並列リクエストを処理 |

---

## 3. `tasks/` — 評価・学習タスク定義

各タスクはデータセットのラッパーで、会話形式 (`{"messages": [...]}`) に変換して提供する。`scripts/` から学習データとしても評価データとしても利用される。

### 基底クラス

| ファイル | 役割 |
|---|---|
| `common.py` | `Task` 基底クラス、`TaskMixture` (複数タスクをシャッフル混合)、`TaskSequence` (順次結合)、`render_mc` (多肢選択フォーマット) を定義 |

### 各タスク

| ファイル | データセット | eval_type | 用途 |
|---|---|---|---|
| `arc.py` | ARC (AI2 Reasoning Challenge) | categorical | 科学的推論の多肢選択。SFT評価・ChatCORE |
| `mmlu.py` | MMLU (57分野) | categorical | 知識の多肢選択。SFT学習データ + 評価 |
| `gsm8k.py` | GSM8K (小学校数学) | generative | Python計算機ツール呼び出しを含む数学問題。SFT学習 + RL報酬 |
| `humaneval.py` | HumanEval (コーディング) | generative | コード生成問題。`execution.py` でコードを実行して正誤判定 |
| `smoltalk.py` | SmolTalk (一般会話) | — | SFT学習の主要データ (train: 460K行) |
| `spellingbee.py` | 合成データ (英単語リスト) | generative | 文字カウント (`SpellingBee`) と単語スペリング (`SimpleSpelling`)。Python ツール使用の練習 |
| `customjson.py` | ローカル JSONL ファイル | — | カスタム会話データの読み込み (identity conversations など) |

---

## フォルダ間の依存関係

```
tasks/
  └─ common.py (Task基底クラス)
  └─ arc.py, mmlu.py, gsm8k.py, ... (各タスク実装)
        ↓ インポート
scripts/
  └─ chat_sft.py  (学習データとして使用)
  └─ chat_eval.py (評価データとして使用)
  └─ chat_rl.py   (報酬計算に使用)
        ↓ インポート
nanochat/
  └─ gpt.py, engine.py, tokenizer.py, ... (コアライブラリ)
```

`nanochat/` は他のどこにも依存しない純粋なライブラリ層、`tasks/` はデータ定義層、`scripts/` はそれらを組み合わせた実行層という3層構造になっている。
