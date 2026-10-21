#!/usr/bin/env bash
# Build the inference env in .venv: python 3.12, requirements.txt, flash-attn from source.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install uv
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
export VIRTUAL_ENV=$PWD/.venv
uv pip install --index-strategy unsafe-best-match -r requirements.txt
uv pip install packaging ninja psutil wheel setuptools

# Must match torch's CUDA (12.6). Not ${CUDA_HOME:-...}: this machine exports CUDA_HOME=/usr/local/cuda (11.7).
export CUDA_HOME=${FA_CUDA_HOME:-/workspace/chicheng/cuda-12.6}
export PATH=${CUDA_HOME}/bin:${PATH}
export FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-80}   # A100 only
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=${MAX_JOBS:-64}
uv pip install --no-build-isolation "flash-attn==2.8.3"

# load_video shells out to ffprobe/ffmpeg; no system ffmpeg here, so put static builds in .venv/bin
uv pip install static-ffmpeg
FF=$(.venv/bin/python -c "import static_ffmpeg.run as r, os; print(os.path.dirname(r.get_or_fetch_platform_executables_else_raise()[0]))")
ln -sf "${FF}/ffmpeg" .venv/bin/ffmpeg
ln -sf "${FF}/ffprobe" .venv/bin/ffprobe

.venv/bin/python -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
