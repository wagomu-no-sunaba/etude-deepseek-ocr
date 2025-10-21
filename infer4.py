# infer.py — DeepSeek-OCR on macOS (Apple Silicon / M-series)
# - CUDA前提のinfer実装をMPS/CPUで動かすためのモンキーパッチ込み
# - sdpa未対応のため attn_implementation="eager"
# - generate() の強制デフォルトで temperature=0.0 問題＆反復を抑制
# - 公式READMEのプロンプトを --preset で選択可能（doc/general/free）
#   ref: DeepSeek-OCR README (Usage / prompts) https://huggingface.co/deepseek-ai/DeepSeek-OCR
#        「Convert the document to markdown.」「OCR this image.」「Free OCR.」

import os
import sys
import argparse
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

MODEL_NAME_DEFAULT = "deepseek-ai/DeepSeek-OCR"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="入力画像のパス（png/jpgなど）")
    parser.add_argument("--model", default=MODEL_NAME_DEFAULT, help="Hugging Face のモデル名/ローカルパス")
    parser.add_argument("--preset", choices=["doc", "general", "free"], default="doc",
                        help="プロンプトのプリセット: doc(既定)/general/free")
    parser.add_argument("--out", default="outputs", help="出力ディレクトリ")
    parser.add_argument("--image-size", type=int, default=512, help="短辺の推論サイズ（重い場合は下げる）")
    parser.add_argument("--base-size", type=int, default=1024, help="内部基準サイズ")
    parser.add_argument("--no-crop", action="store_true", help="crop_modeを無効化")
    args = parser.parse_args()

    IMAGE_PATH = args.image
    OUT_DIR = args.out
    MODEL_NAME = args.model

    # ログ静音（必要なら warning/info に）
    hf_logging.set_verbosity_error()

    # ===== デバイス判定 =====
    use_mps = torch.backends.mps.is_available()
    DEVICE = "mps" if use_mps else "cpu"

    # ===== CUDAハードコード回避のモンキーパッチ =====
    # infer() が .cuda() と autocast('cuda') を呼ぶため、MPS/CPUへ迂回
    def _tensor_cuda(self, *a, **k):
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
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",   # DeepseekOCRはsdpa非対応
        trust_remote_code=True,
        use_safetensors=True,
    ).eval()

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if use_mps:
        model.to("mps")
        # 動作安定確認後、必要に応じて半精度を検討:
        # model.to(torch.float16)

    # ===== generate() をパッチ：infer()内部の固定引数を上書き =====
    _orig_generate = model.generate
    def _patched_generate(*g_args, **g_kwargs):
        # infer() 側が do_sample=False / temperature=0.0 を渡してくるため、
        # ここでサンプリング系を強制上書きして反復やエラーを回避
        g_kwargs.setdefault("do_sample", True)
        if g_kwargs.get("temperature", 0.0) == 0.0:
            g_kwargs["temperature"] = 0.2     # 低温度で暴走を抑制
        g_kwargs.setdefault("top_p", 0.9)
        g_kwargs.setdefault("repetition_penalty", 1.15)  # 反復抑制やや強め
        g_kwargs.setdefault("no_repeat_ngram_size", 4)
        g_kwargs.setdefault("max_new_tokens", 1200)
        g_kwargs.setdefault("pad_token_id", getattr(model.config, "pad_token_id", None))
        return _orig_generate(*g_args, **g_kwargs)
    model.generate = _patched_generate

    # ===== プロンプト（README掲載の3種をプリセット化） =====
    # ref: README Usage — "<image>\n<|grounding|>Convert the document to markdown."
    #                         "<image>\n<|grounding|>OCR this image."
    #                         "<image>\nFree OCR."
    if args.preset == "doc":
        instr = "Convert the document to markdown."
    elif args.preset == "general":
        instr = "OCR this image."
    else:  # free
        instr = "Free OCR."

    # 余計な説明を避けるための追加指示（日本語向け調整）
    keep_clean = (
        "Return ONLY the extracted text. No descriptions of background, layout or colors. "
        "If tables exist, format them as Markdown tables. Avoid word repetition. "
        "If the text is Japanese, keep Japanese as is."
    )

    prompt = f"<image>\n<|grounding|>{instr} {keep_clean}"

    # ===== 実行 =====
    os.makedirs(OUT_DIR, exist_ok=True)

    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=IMAGE_PATH,
        output_path=OUT_DIR,
        base_size=args.base_size,
        image_size=args.image_size,
        crop_mode=not args.no_crop,   # 既定: True（=クロップ有効）
        save_results=True,
        test_compress=False,          # 揺らぎを減らす
    )

    print("✅ Done. Results under:", os.path.abspath(OUT_DIR))
    print("   Preset:", args.preset, "| Device:", DEVICE)

if __name__ == "__main__":
    main()

