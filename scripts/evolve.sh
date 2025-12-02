python -m src.evolution.runner \
  --train-config configs/baseline.yaml \
  --controller-config configs/controller.yaml \
  --evolve-config configs/evolve.yaml \
  --outdir runs/evolve_long \
  --generations 1