# ANT-MNIST

**90.4% MNIST accuracy using integer-only arithmetic, mastery learning, and frozen random spatial features. No floats. No GPU. Pure Rust on CPU.**

This is Blackfall Labs' proving ground for signal-native machine learning — proving that ternary signal systems can learn real-world classification tasks without floating-point arithmetic or traditional gradient descent.

---

## Results

| System | Test Accuracy | Weights | Learning | Time/Epoch |
|--------|--------------|---------|----------|------------|
| f32 LeNet-5 (backprop, all layers) | **98.8%** | 44,426 (all plastic) | SGD | 32s |
| **8-pair dual hemisphere ensemble** | **90.4%** | 710K (13.6K plastic) | Gradient-mastery-thermal | 153s |
| 5-pair dual hemisphere | 89.9% | 444K (8.5K plastic) | Gradient-mastery-thermal | 112s |
| Gradient-mastery hybrid (10v) | 89.5% | 444K (8.5K plastic) | Gradient-mastery | 58s |
| 15-voter mastery | 86.4% | 666K (12.7K plastic) | Pure mastery | 94s |
| 10-voter mastery | 84.0% | 444K (8.5K plastic) | Mastery + thermal step | 72s |
| Single mastery | 67.4% | 44K (850 plastic) | Pure mastery | 16s |
| P×M×K basin learning | 31.7% | 44K (840 plastic) | Basin update | 23s |
| Flow weights (single layer) | 13.2% | 7,840 | Momentum-based | ~100ms |

---

## Architecture

```
Input (28×28 pixels, i16)
  → Conv1 (6 × 5×5, FROZEN random i16) → ReLU → AvgPool 2×2
  → Conv2 (16 × 6×5×5, FROZEN random i16) → ReLU → AvgPool 2×2
  → FC1 (256→120, FROZEN random i16) → ReLU
  → FC2 (120→84, FROZEN random i16) → ReLU
  → FC3 (84→10, GRADIENT-MASTERY LEARNING) ← only this layer learns
```

### Dual Hemisphere

Each pair consists of two networks with:
- **Different random seeds** — different spatial feature extraction
- **Different pressure thresholds** (8 vs 12) — different learning sensitivity
- **Combined output** — full sum of both hemispheres, no information loss

Multiple pairs are ensembled by summing their combined outputs. The class with the highest total score wins.

### Weight Representation

```
MasteryWeight = [polarity: i8, magnitude: u8, temperature: u8, pressure: i16]
                 5 bytes per weight
```

- **Polarity × Magnitude** = signal value (the weight)
- **Temperature** = stability tracker (255=HOT/plastic, 0=COLD/stable)
- **Pressure** = gradient accumulator (resets on commit)

### Learning Rule: Gradient-Mastery-Thermal Hybrid

A fusion of three mechanisms:

1. **Gradient signal**: Softmax cross-entropy gradient computed in i32. Provides precise error direction. No backpropagation through frozen layers — only the output layer gradient is needed.

2. **Mastery accumulation**: Gradient is accumulated into each weight's `pressure` field across samples. Weight only updates when `|pressure| >= threshold`. This acts as a natural mini-batch — smoothing per-sample noise and preventing oscillation.

3. **Thermal-adaptive step size**: When pressure crosses threshold, the magnitude of the update is capped by the weight's temperature band:
   - HOT (temp >= 192): max step ±8 (coarse alignment, early learning)
   - WARM (temp >= 128): max step ±5
   - COOL (temp >= 64): max step ±3 (fine tuning)
   - COLD (temp < 64): max step ±1 (precision adjustment)

Temperature cools by 1 on correct classifications, warms by 3 on errors. This creates per-weight adaptive learning rate without any floating-point computation.

### Why Frozen Random Features Work

Convolutional layers and FC1/FC2 are initialized with random integer weights and never updated. This works because:

1. **Random projections create class separation** — even random spatial filters detect edges, textures, and patterns that vary across digit classes (Johnson-Lindenstrauss lemma)
2. **The output layer only needs to learn alignment** — which random features matter for which class
3. **No gradient truncation** — the problem with i16 backprop through all layers was bit-shift precision loss. With only one plastic layer, gradient precision is maintained

---

## Key Findings

### What works

| Technique | Contribution |
|-----------|-------------|
| Frozen random conv/FC layers | Foundation — provides feature separation without learning |
| Gradient-mastery hybrid | +5.5% over pure mastery (89.5 vs 84.0) |
| Dual hemispheres | +0.4% and much better single-pair accuracy (87.6 vs 67.4) |
| Thermal-adaptive step size | +0.1% and significantly better epoch stability |
| Ensemble (more voters) | +6% from 3 to 15 voters, diminishing returns above 10 |
| Normalized confidence voting | Matches majority vote but more stable over epochs |

### What doesn't work

| Technique | Result | Why |
|-----------|--------|-----|
| Learning hidden layers | 65% to 15% | Destabilizes features the output depends on |
| Alternating freeze | 67% to 14% to 21% | Hidden learning corrupts features permanently |
| i16 backprop through all layers | 10% (random) | Gradient dies from bit-shift precision loss |
| P×M×K multiplier in weights | 84% to 35% | Changes effective values, disrupts frozen feature pipeline |
| Temperature gating learning | 84% to 79% | Dampens gradients before weights differentiate |
| Temperature in forward pass | 84% to 79-83% | All weights start hot, any modification hurts epoch 1 |
| Lateral inhibition (suppression) | 90% to 83% | Confident wrong hemisphere destroys correct signal |
| Raw score voting | 84% to 79% | Uncalibrated voters with big scores dominate |
| Margin-weighted voting | 84% to 81% | Confident wrong voters dominate |
| Single-layer classifier (any learning rule) | 10-13% | Pixel-level signal too noisy for 10-class discrimination |

### Temperature's role (proven)

- **Forward pass**: NO — clean signal, no interference
- **Step size**: YES — per-weight adaptive learning rate (HOT=8, COLD=1)
- **Monitoring**: YES — cool on correct, warm on error (free observability)

---

## Confusion Analysis

Top digit confusions at 84% accuracy (10-voter mastery):

```
4 -> 9 : 103 errors    (vertical stroke + top curve overlap)
9 -> 4 :  76 errors
7 -> 9 :  62 errors    (diagonal stroke confusion)
8 -> 5 :  62 errors    (loop vs curve)
3 -> 8 :  60 errors    (similar curves)
```

These confusions come from the frozen random conv filters not distinguishing specific spatial features (the difference between a 4's crossing and a 9's curve). The f32 version learns these distinctions through backprop; our system compensates with ensemble diversity.

### Voter agreement analysis (10 voters)

- 9.5% of samples: all voters agree (99.8% correct when unanimous)
- 84.0%: majority correct
- 14.1%: majority wrong but at least one voter correct (opportunity)
- 1.9%: all voters wrong (hard ceiling at 190 samples)
- Voter agreement: 50-60% pairwise (genuine diversity)

---

## Running

```bash
# f32 baseline
cargo run --release --bin lenet-experiment -- data

# 10-voter mastery ensemble
cargo run --release --bin lenet-ensemble -- data 10 3

# 8-pair dual hemisphere (best result)
cargo run --release --bin lenet-dual-hemi -- data 8 3

# Voter disagreement analysis
cargo run --release --bin voter-analysis -- data
```

MNIST data files go in `data/`:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

---

## Resource Profile (8-pair dual hemisphere)

| Metric | Value |
|--------|-------|
| Test accuracy | 90.4% |
| Total weights | 710,816 |
| Plastic weights | 13,600 (output layers only) |
| Weight memory (i16) | 1.39 MB |
| Weight memory (5-byte MasteryWeight) | 3.47 MB |
| Process RSS | 96.5 MB (includes MNIST data) |
| Training time | 153s/epoch (16 voters on 12 threads via rayon) |
| Per-weight size | 5 bytes (polarity + magnitude + temperature + pressure) |

---

## File Structure

```
src/
  lenet_f32.rs           -- f32 LeNet-5 baseline (98.8%)
  lenet_mastery.rs       -- MasteryWeight + MasteryLayer + LeNetMastery
  lenet_i16.rs           -- P*M*K signal weight variant (31.7%)
  lenet_ensemble.rs      -- N-voter ensemble with rayon
  lenet_dual_hemi.rs     -- Dual hemisphere pairs (90.4%)
  lenet_thermal.rs       -- ThermalWeight from atomic-neural-transistors
  lenet_sweep.rs         -- Multi-variant sweep runner
  lenet_experiment.rs    -- Original dual experiment runner
  flow.rs                -- FlowWeight momentum prototype (13.2%)
  flow_experiment.rs     -- Flow weight experiment runner
  voter_analysis.rs      -- Voter disagreement analysis
  viz.rs                 -- WebSocket dashboard (detector architecture)
  mnist.rs               -- MNIST IDX file parser
  main.rs                -- Original CLI trainer

experiments/             -- CSV results and logs from all runs
dashboard/               -- React 19 + Vite + Tailwind dashboard
runes/                   -- Runes scripts for detector architecture
```

---

## What This Proves

1. **Signal-native ML works.** Integer-only arithmetic can learn spatial classification on a standard benchmark. 90.4% on MNIST with no floats.

2. **Frozen random projections are powerful.** You don't need to learn conv filters. Random spatial filters provide sufficient feature separation. Learning only needs to happen on the final classification layer.

3. **Mastery learning is competitive.** Pressure-gated gradient accumulation with thermal-adaptive step sizes matches or exceeds naive integer SGD. The hysteresis prevents oscillation that kills raw per-sample updates.

4. **Temperature has a real role.** Not as a gate (kills learning), but as a per-weight adaptive learning rate. Cold weights make precise adjustments. Hot weights make bold moves.

5. **Dual hemispheres beat ensembles.** Two networks with different sensitivity, combined by simple addition, outperform the same number of independent voters. Complementary processing beats redundant processing.

6. **The ceiling is in the features.** Every experiment confirms: voting tricks, learning rule improvements, and architectural variations all plateau at the same ceiling. The remaining 8.4-point gap to f32 lives in the frozen random features vs learned conv filters.

---

*Blackfall Labs — Neuromorphic systems. Not machine learning. But when we do ML, we do it in integers.*
