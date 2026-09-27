#!/usr/bin/env bash
# Запуск веб-морды YuE2-3B: по умолчанию на всех интерфейсах (домашняя сеть),
# изнутри — http://127.0.0.1:8220, с устройств LAN — http://<IP-машины>:8220
# YUE2_HOST=127.0.0.1 ./run.sh — только локально
# YUE2_VAE=m-a-p/YuE2-Vae-legacy ./run.sh — VAE для бенчмарков (по умолчанию YuE2-Vae)
set -euo pipefail
cd "$(dirname "$0")"
export YUE2_HOST="${YUE2_HOST:-0.0.0.0}"
export YUE2_PORT="${YUE2_PORT:-8220}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec .venv/bin/python -m uvicorn server:app --host "$YUE2_HOST" --port "$YUE2_PORT" --timeout-keep-alive 120
