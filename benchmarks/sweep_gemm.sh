#!/usr/bin/env bash
set -e

OUTDIR=build
mkdir -p $OUTDIR

# 벡터 크기별로 native/simd 벤치 반복
for SIZE in 64 128 256 512; do
  echo "== SIZE=${SIZE} =="
  for BACKEND in native simd; do
    echo "[backend=$BACKEND]" | tee -a $OUTDIR/bench_${BACKEND}.log
    MINIDL_MATMUL_BACKEND=$BACKEND \
      ./bin/bench_gemm --M $SIZE --K $SIZE --N $SIZE --threads 8 \
      | tee -a $OUTDIR/bench_${BACKEND}.log
  done
done
