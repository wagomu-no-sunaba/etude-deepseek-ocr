from transformers import AutoModel, AutoTokenizer
import torch, os, sys

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
OUT_DIR = "outputs"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# macOS/MPS では flash-attn を使わずに sdpa/eager を選択
ATTN_IMPL = "eager"   # or "eager"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    _attn_implementation=ATTN_IMPL,
    trust_remote_code=True,
    use_safetensors=True,
)

model = model.eval()

if device == "mps":
    model = model.to("mps")
    dtype = torch.float16
else:
    dtype = torch.float32

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
os.makedirs(OUT_DIR, exist_ok=True)

# DeepSeek-OCR の infer API を呼ぶ（READMEの使用例をMPS向けに調整）
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

