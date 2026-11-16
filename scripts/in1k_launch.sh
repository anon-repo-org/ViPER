#!/bin/bash
# ImageNet-1k runs (DeiT-S, 100 epochs, global batch 1024, lr 1e-3), launched outside the queue.
#   scripts/in1k_launch.sh <method> <gpus, e.g. 5,6> [port] [seed]
# Global batch is kept at 1024 for any GPU count: bs 256 per GPU, accum = 4 / #GPUs.
set -e
m=$1; g=$2; port=${3:-29560}; seed=${4:-0}
n=$(echo "$g" | tr ',' '\n' | wc -l); accum=$((4 / n))
case $m in
  viper)        pe=viper; kw='{"lepe": true}' ;;
  abl_no_gates) pe=viper; kw='{"lepe": true, "use_gate": false}' ;;
  viper_glr)    pe=viper; kw='{"lepe": true}'; extra="--gate_lr_mult 0.125" ;;   # gate branch at the IN-100 lr
  *)            pe=$m;    kw='{}' ;;
esac
out=results/in1k_deit_small/${m}_s${seed}
mkdir -p "$out" runs
CUDA_VISIBLE_DEVICES=$g setsid nohup python -m torch.distributed.run --nproc_per_node "$n" --master_port "$port" \
  -m viper.train --dataset imagenet1k --backbone deit_small --epochs 100 --warmup 5 --bs 256 --accum "$accum" \
  --workers 10 --pe "$pe" --pe_kwargs "$kw" ${extra:-} --seed "$seed" --out "$out" > "runs/in1k_${m}_s${seed}.out" 2>&1 < /dev/null &
echo "launched $m on GPUs $g (n=$n, accum=$accum) -> $out"
