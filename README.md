# DeepSeek-OCR for macOS (Apple Silicon)

Apple Silicon Mac 上で DeepSeek-OCR を動かすための最小構成ガイドです。  
画像内の文書を OCR で Markdown に変換します。

## 前提条件
- macOS（Mシリーズチップ対応）
- Nix（flakes 有効化済み）
- direnv インストール済み

## セットアップ
```bash
direnv allow
uv venv
source .venv/bin/activate
uv sync
````

初回実行時に `deepseek-ai/DeepSeek-OCR` モデルが自動ダウンロードされます（数GB）。

## 使い方

```bash
python infer.py image.png
# または
uv run python infer.py image.png
```

結果は `outputs/` に保存されます。

## 技術構成

* Python 3.12+
* PyTorch 2.6.0（MPS対応）
* transformers 4.46.3
* uv（パッケージ管理）
* Nix flakes（環境管理）

環境変数 `PYTORCH_ENABLE_MPS_FALLBACK=1` により、未対応演算は自動的に CPU にフォールバックします。

## トラブルシューティング

* MPSエラー時：`infer.py` の `device = "mps"` を `"cpu"` に変更
* モデルダウンロード失敗時：ネットワーク確認後に再実行

## キャッシュ削除

モデルや一時ファイルを削除する場合：

```bash
rm -rf ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR
rm -rf ~/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR
```

Hugging Face 全体を削除：

```bash
rm -rf ~/.cache/huggingface
```

仮想環境を削除：

```bash
rm -rf .venv
```

Nix キャッシュ削除：

```bash
nix-collect-garbage -d
```

キャッシュを無効化して実行：

```bash
export TRANSFORMERS_CACHE=/tmp/hf_cache
```

## 参考

* DeepSeek-OCR: [https://github.com/deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)

