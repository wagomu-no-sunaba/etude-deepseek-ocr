# infer.py (macOS / MPS 用の回避版)
from transformers import AutoModel, AutoTokenizer
import torch, os, sys

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "image.png"
OUT_DIR = "outputs"

# ---- デバイス判定 ----
use_mps = torch.backends.mps.is_available()
DEVICE = "mps" if use_mps else "cpu"

# ---- ★モンキーパッチ：CUDA 前提コードを MPS/CPU に迂回 ----
# 1) Tensor.cuda() → to(DEVICE)
def _tensor_cuda(self, *args, **kwargs):
    return self.to(DEVICE)
torch.Tensor.cuda = _tensor_cuda  # type: ignore[attr-defined]

# 2) autocast('cuda', ...) → autocast('mps' or 'cpu', ...)
from torch.amp.autocast_mode import autocast as _autocast_impl
def _autocast(device_type=None, **kwargs):
    if device_type == "cuda":
        device_type = "mps" if DEVICE == "mps" else "cpu"
    return _autocast_impl(device_type=device_type, **kwargs)
torch.amp.autocast_mode.autocast = _autocast  # type: ignore[attr-defined]
torch.autocast = _autocast  # 互換

# 3) is_available() に依存した分岐を抑止（任意）
torch.cuda.is_available = lambda: False  # type: ignore[assignment]

# ---- モデル読込（sdpa不可なので eager を明示） ----
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    trust_remote_code=True,
    use_safetensors=True,
).eval()

if use_mps:
    model.to("mps")

# ---- 推論 ----
os.makedirs(OUT_DIR, exist_ok=True)
prompt = "<image>\n<|grounding|>Convert the document to markdown. "

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=IMAGE_PATH,
    output_path=OUT_DIR,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True,
)

print("Done. Results under:", OUT_DIR)

