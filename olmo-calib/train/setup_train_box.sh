#!/bin/bash
# Set up a vast box for the DPO reruns: open-instruct (pinned commit, its own uv venv) + our eval scripts + data.
# Needs an NVIDIA driver >= 570 (CUDA 12.8). Use a plain CUDA/PyTorch template, not the vLLM one (that one starts a
# vLLM server holding ~88GB per GPU; if you do use it, stop it first: supervisorctl stop vllm).
# Run from /workspace after copying olmo-calib/ there:  bash olmo-calib/train/setup_train_box.sh
set -euo pipefail
OI_COMMIT=11826255077617a46919ce75cadf1f3d53f30dac   # open-instruct main, 2026-09-30 (the code we read)
export WS=${WS:-/workspace}
cd $WS

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h $WS | tail -1
free -g | head -2

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

if [ ! -d open-instruct ]; then
  git clone https://github.com/allenai/open-instruct.git
fi
cd open-instruct
git fetch -q origin $OI_COMMIT 2>/dev/null || true
git checkout -q $OI_COMMIT
# Same command as open-instruct's Dockerfile: frozen lockfile, CUDA 12.8 wheels (torch, vllm, flash-attn, deepspeed).
# The dev group is needed too: open_instruct/utils.py imports beaker (beaker-py) unconditionally.
uv sync --frozen --no-default-groups --group dev --group cuda12
# The lockfile pins transformers 5.4.0, which computes Olmo 3 wrongly (likely its YaRN rope handling; it warns
# "beta_fast must be a float"): on 64 MMLU questions only 77% of answers match vLLM / transformers 5.17, and full
# MMLU accuracy of Instruct-SFT drops 0.609 -> 0.549. Training uses the same HF model class, so upgrade.
# Run from /tmp: inside the project dir uv applies the project's settings and silently keeps 5.4.0.
(cd /tmp && uv pip install --python $WS/open-instruct/.venv/bin/python "transformers==5.17.0")
.venv/bin/python -c "import transformers; assert transformers.__version__ == '5.17.0', transformers.__version__"
.venv/bin/python $WS/olmo-calib/train/patch_open_instruct.py $WS/open-instruct
# ZeRO-1 config (open-instruct ships only stage 2/3). ZeRO-2 reduces the full gradient after every micro-batch;
# ZeRO-1 once per step: 1.4-1.6x faster on 2xB200 at batch 1 per GPU, same optimizer math.
sed 's/"stage": 2/"stage": 1/' configs/ds_configs/stage2_no_offloading_accelerate.conf > configs/ds_configs/stage1_no_offloading_accelerate.conf
.venv/bin/python -c "import open_instruct.dpo_utils"   # fails fast on missing imports (it pulls in most of the package)
.venv/bin/python -c "import nltk; [nltk.download(r, quiet=True) for r in ('punkt', 'punkt_tab')]"   # IFEval's checker splits sentences
.venv/bin/python -c "import torch, transformers, deepspeed, vllm, flash_attn; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| deepspeed', deepspeed.__version__, '| vllm', vllm.__version__, '| gpus', torch.cuda.device_count())"

# Training sets (Instruct + Think DPO, ~2.5 GB download) -> /workspace/dpo_data
cd $WS/olmo-calib/train
[ -f $WS/dpo_data/summary.json ] || HF_HUB_ENABLE_HF_TRANSFER=1 $WS/open-instruct/.venv/bin/python make_dpo_data.py --out $WS/dpo_data
echo SETUP_OK
