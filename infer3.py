# infer.py — DeepSeek-OCR on macOS (Apple Silicon / M-series)
# - CUDA前提のinfer実装をMPS/CPUで動かすためのモンキーパッチ込み
# - sdpa未対応のため attn_implementation="eager"
# - generate() に強制デフォルトを噛ませ、temperature=0.0問題を回避

import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

# ===== 基本設定 =====
MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "image.png"
OUT_DIR = "outputs"

hf_logging.set_verbosity_error()  # ログ抑制（必要に応じて下げる）

# ===== デバイス判定 =====
use_mps = torch.backends.mps.is_available()
DEVICE = "mps" if use_mps else "cpu"

# ===== CUDAハードコード回避のモンキーパッチ =====
def _tensor_cuda(self, *args, **kwargs):
    return self.to(DEVICE)
torch.Tensor.cuda = _tensor_cuda  # type: ignore[attr-defined]

from torch.amp.autocast_mode import autocast as _autocast_impl
def _autocast(device_type=None, **kwargs):
    if device_type == "cuda":
        device_type = "mps" if DEVICE == "mps" else "cpu"
    return _autocast_impl(device_type=device_type, **kwargs)
torch.amp.autocast_mode.autocast = _autocast  # type: ignore[attr-defined]
torch.autocast = _autocast

# CUDA分岐を踏ませない（保険）
torch.cuda.is_available = lambda: False  # type: ignore[assignment]

# ===== トークナイザ & モデル読込 =====
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# PAD未設定による警告・挙動不定を防止（EOSを流用）
if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModel.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",   # DeepseekOCRはsdpa未対応
    trust_remote_code=True,
    use_safetensors=True,
).eval()

# モデル側にもpad_token_idを反映
if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id

# デバイスへ
if use_mps:
    model.to("mps")
    # 安定確認後に必要なら半精度を検討
    # model.to(torch.float16)

# ===== generate() をパッチ：infer() 内の固定引数を上書き =====
_orig_generate = model.generate
def _patched_generate(*args, **kwargs):
    # infer() 側が do_sample=False / temperature=0.0 にするため、こちらで強制上書き
    kwargs.setdefault("do_sample", True)
    # temperature は > 0 が必須
    if kwargs.get("temperature", 0.0) == 0.0:
        kwargs["temperature"] = 0.7
    kwargs.setdefault("top_p", 0.9)
    kwargs.setdefault("repetition_penalty", 1.08)
    kwargs.setdefault("no_repeat_ngram_size", 3)
    kwargs.setdefault("max_new_tokens", 2048)
    # pad_token 未設定での警告回避
    kwargs.setdefault("pad_token_id", getattr(model.config, "pad_token_id", None))
    return _orig_generate(*args, **kwargs)
model.generate = _patched_generate  # モデルに差し替え

# ===== プロンプト（OCR特化） =====
prompt = (
    "<image>\n"
    "<|grounding|>"
    "あなたはOCRエンジンです。画像から文字だけを正確に抽出してください。"
    "出力はクリーンなMarkdown。背景や色・レイアウトの説明は書かないでください。"
    "同じ語の繰り返しを避け、表はMarkdown表に整形してください。"
)

# ===== 実行 =====
os.makedirs(OUT_DIR, exist_ok=True)

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=IMAGE_PATH,
    output_path=OUT_DIR,
    base_size=1024,
    image_size=512,       # 重ければさらに下げる（例: 448）
    crop_mode=True,
    save_results=True,
    test_compress=False,  # 揺らぎを減らす
)

print("✅ Done. Results under:", os.path.abspath(OUT_DIR))
