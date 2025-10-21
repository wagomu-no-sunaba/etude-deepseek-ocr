{
  description = "DeepSeek-OCR dev shell (macOS Apple Silicon) with uv + Python";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            uv
            ruff
            python312
            pkg-config
            cmake
            ninja
            llvmPackages.clang
            rustc cargo
          ];

          # PyTorch/MPS 用の環境変数（安定動作用）
          # MPSが未実装の演算は自動でCPUにフォールバック
          # （必要に応じて外してください）
          shellHook = ''
            export PYTORCH_ENABLE_MPS_FALLBACK=1
            echo "Dev shell ready. Python: $(python3 --version); uv: $(uv --version 2>/dev/null || true)"
            echo "Tip: create & activate venv with:  uv venv && source .venv/bin/activate"
          '';
        };
      });
}

