#!/bin/bash
# Wait for a genuinely complete shard, then start training alongside the pool
# build. "Complete" means both the COMPLETE marker AND a full manifest -- the
# marker alone was not enough once, when an orphaned generator recreated a
# shard that a second writer then truncated.
cd /home/pranesh/Documents/semicon/semicon-driftsense
EXPECT=8001   # header + 1000 canvases x 8 crops
while true; do
  for d in data/pool_p2/s*/; do
    [ -f "$d/COMPLETE" ] || continue
    n=$(wc -l < "$d/manifest.csv" 2>/dev/null || echo 0)
    [ "$n" -ge "$EXPECT" ] && { echo "$(date +%H:%M) starting training on $d ($n rows)"; break 2; }
  done
  sleep 60
done
exec ./venv-train/bin/python train.py \
  --train-dirs data/pool_p2 --refresh-pool --samples-per-epoch 16000 --phase2 \
  --crop 512 --batch-size 8 --epochs 24 --lr 3e-4 --workers 4 --amp \
  --resume weights/driftsense.pt --finetune \
  --val-dir data/val_p2 --val-limit 60 --keep-epochs \
  --out weights/driftsense_p2.pt
