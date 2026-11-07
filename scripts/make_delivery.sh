#!/usr/bin/env bash
# Пересобрать всё в results/delivery/ по текущему checkpoints/best.pt. Строго последовательно.
#   scripts/make_delivery.sh [DEVICE=cpu] [HEROES=auto] [TEST_QUADS=32] [REFINE=50]
# Основной режим сдачи — сеть + REFINE шагов уточнения (nn_refine), быстрый — чистая сеть (nn).
# HEROES=auto: 2 набора с лучшей селективностью основного режима (scripts/selectivity.py) + rocket_smoke
# как пример «похожих целей».
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEVICE=${1:-cpu}
HEROES=${2:-auto}
TEST_QUADS=${3:-32}
REFINE=${4:-50}
OUT=results/delivery

for d in data/demo/*/; do
  name=$(basename "$d")
  for mode in nn nn_refine; do
    sub=$mode; steps=()
    if [ "$mode" = nn_refine ]; then sub=nn_refine$REFINE; steps=(--steps "$REFINE"); fi
    dst="$OUT/demo/$name/$sub"
    mkdir -p "$dst"
    $PY synth.py "$d"target_{1,2,3,4}.png --out "$dst" --method $mode ${steps[@]+"${steps[@]}"} \
        --device "$DEVICE" > "$dst/synth.txt" 2>&1
    $PY evaluate.py "$dst/doe.png" "$d"target_{1,2,3,4}.png --check-recon "$dst" \
        --json "$dst/evaluation.json" > "$dst/evaluation.txt"
  done
  echo "demo $name: ok"
done
$PY scripts/demo_table.py --refine "$REFINE"
SEL=$($PY scripts/selectivity.py --top 2 --refine "$REFINE" | tail -1)
if [ "$HEROES" = auto ]; then HEROES="$SEL rocket_smoke"; fi
for h in $HEROES; do
  $PY scripts/focus_sweep.py --run "$OUT/demo/$h/nn_refine$REFINE" --out "$OUT/focus_sweep_$h.gif"
  $PY scripts/crosstalk.py   --run "$OUT/demo/$h/nn_refine$REFINE" --out "$OUT/crosstalk_$h.png"
done
$PY scripts/compare_methods.py --quads demo --device "$DEVICE" --out "$OUT/methods" --refine-steps "$REFINE"
if [ "$TEST_QUADS" -gt 0 ]; then
  $PY scripts/compare_methods.py --quads "test:$TEST_QUADS" --device "$DEVICE" --out "$OUT/methods_test" \
      --speed-plot "$OUT/speed_quality_test.png" --no-grid --refine-steps "$REFINE"
fi
$PY scripts/update_readme_results.py
