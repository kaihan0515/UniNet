#!/usr/bin/env bash
# Clean single 256px run. Waits for the GPU to free up, then trains + tests once.
cd "/d/111370211/MVA/final/UniNet" || exit 1
source activate MVA_py310_cu121 2>/dev/null || conda activate MVA_py310_cu121 2>/dev/null

# reduce GPU memory fragmentation (helps the memory-heavy eval phase stay stable)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[wait] waiting for GPU to free up..."
for i in $(seq 1 1080); do            # up to ~6 h
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$used" -lt 1500 ]; then
    echo "[wait] GPU free (${used} MiB) @ $(date +%H:%M:%S)"
    break
  fi
  sleep 20
done

echo "######## CLEAN 256px RUN @ $(date +%H:%M:%S) ########"
rm -rf ckpts/SteelBall saved_results/SteelBall
PYTHONUNBUFFERED=1 python main.py --dataset SteelBall --setting oc \
    --epochs 100 --batch_size 4 --image_size 256 --center_crop 256
echo "######## DONE @ $(date +%H:%M:%S) ########"
