# evo-hparam-controller
# Evolutionary Hyperparameter Controller

Black-box evolutionary optimization for learning state-dependent hyperparameter schedules. This work checks whether evolutionary strategies can discover adaptive controllers that outperform static schedules and hand-tuned heuristics.

## Overview

This repository implements and benchmarks evolutionary algorithms for optimizing hyperparameter controllers, parameterized policies that adapt learning rates, momentum, and weight decay based on training state (loss, gradients, epoch progression).

**Benchmark**: CIFAR-10 classification with ResNet-20  
**Controller Input**: Training metrics (loss, gradient norm, epoch fraction)  
**Controller Output**: Per-iteration hyperparameter adjustments  
**Optimization**: Population-based methods with fitness evaluation via training performance

## Key Research Questions

- Can evolutionary methods discover effective adaptive hyperparameter policies without gradient information?
- How do learned controllers compare to static schedules (cosine annealing, step decay) and meta-learning approaches?
- What is the computational cost trade-off between controller search and performance gains?
- Do evolved controllers generalize across different architectures or datasets?
