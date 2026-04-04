# Experiment A Results: Flow Weights Single-Layer 784→10

## Date: 2026-04-03

## Configuration
- Architecture: 784 → 10 (single FlowMatrix, no hidden layer)
- FlowWeight: 6 bytes (value: i16, momentum: i16, volatility: u16)
- Weight count: 7,840 (10 rows × 784 cols)
- Weight memory: 47,040 bytes (45.9 KB)
- Process RSS: 96.5 MB (includes MNIST data)
- Training: 100 cycles × 200 samples/digit (2000 samples/cycle)
- Eval: 1000 test samples every 5 cycles
- Targets: +40 for correct class, -20 for all others
- Momentum decay: 230/256 (0.898)
- Volatility decay: 250/256 (0.977)
- Value range: ±500
- Nudge scale: error × input / 100, clamped ±500
- Momentum-to-value divisor: /4

## Timing
| Phase | Time |
|-------|------|
| Cycle 1 | 94ms |
| Cycle 25 | 514ms |
| Cycle 50 | 525ms |
| Cycle 76 | 105ms (reset?) |
| Cycle 100 | 518ms |

Cycle time grows ~5× from start to ~25 cycles, then appears to reset periodically. Pattern correlates with which digit dominates — possibly the training set ordering causes cache effects.

## Accuracy
| Metric | Value |
|--------|-------|
| Best test accuracy | 13.2% (cycle 35) |
| Worst test accuracy | 4.5% (cycle 75) |
| Mean test accuracy | ~9.5% |
| Baseline (random) | 10% |
| Train accuracy | 8-12% (never exceeds 12.4%) |

**The system does not learn.** Test accuracy oscillates around random baseline.

## Mode Collapse
| Cycle | Dominant class | Collapse score | Per-class |
|-------|---------------|----------------|-----------|
| 1 | d9: 89% | 0.80 | all others 0% |
| 5 | d6: 100% | 0.90 | all others 0% |
| 10 | d3: 79% | 0.70 | d4: 10% |
| 15 | d6: 55%, d7: 62% | 0.50 | first multi-class moment |
| 20 | d3: 99% | 0.89 | back to single |
| 25 | d0: 100% | 0.90 | rotated |
| 30 | d4: 68%, d7: 43% | 0.57 | brief multi |
| 35 | d4: 49%, d7: 30%, d8: 54% | 0.41 | best spread |
| 40 | d7: 100% | 0.90 | collapsed again |
| 50 | d8: 100% | 0.90 | — |
| 60 | d2: 24%, d3: 7%, d4: 35% | 0.28 | lowest collapse |
| 70 | d3: 100% | 0.90 | — |
| 80 | d8: 72% | 0.63 | d1/d2/d5 showing |
| 90 | d4: 100% | 0.90 | — |
| 100 | d0: 100% | 0.90 | — |

Mode collapse rotates through digits every ~5-10 cycles. The system never sustains multi-class discrimination.

## Weight Statistics
| Metric | Cycle 1 | Cycle 50 | Cycle 100 |
|--------|---------|----------|-----------|
| Magnitude mean | 241 | 275 | 278 |
| Magnitude std | 182 | 177 | 189 |
| Momentum mean | 371 | 506 | 370 |
| Nonzero weights | 7831 | 7839 | 7837 |
| HOT weights | 7840 | 7840 | 7840 |
| WARM weights | 0 | 0 | 0 |
| COOL weights | 0 | 0 | 0 |
| COLD weights | 0 | 0 | 0 |

**All weights remain HOT for the entire experiment.** Volatility never decreases because momentum never settles — mode collapse keeps the system in perpetual oscillation. No weight achieves stability.

## Diagnosis

### Why it fails

1. **Target asymmetry**: correct class gets +40, nine wrong classes each get -20. Total negative signal per sample = 9 × 20 = 180 vs positive = 40. The system is 4.5× more motivated to suppress than to activate. Over time, all outputs get driven negative. Whichever class is least negative "wins" by default, not by learned discrimination.

2. **No hidden structure**: 784 pixels → 10 classes with a single linear layer cannot learn MNIST. The digits share too much pixel-level structure (edges, center mass). The discriminative signal at the pixel level is tiny compared to shared structure. A hidden layer is needed to extract features before classification.

3. **No input normalization**: pixel signals range from -60 to +60 but most pixels are 0 (background). The few active pixels dominate the dot product, and those pixels are often in the same positions across digits.

### What the weights see

Each of the 10 output rows is a 784-element template. Row 3 (for digit "3") should ideally look like a "3" — positive weights where 3s have bright pixels, negative where they don't. But because the training signal is dominated by "not X" pressure (9× per sample), the templates converge to "NOT whatever is currently winning" rather than "IS my digit."

### Comparison to ThermalWeight baseline

| Metric | ThermalWeight | FlowWeight |
|--------|--------------|------------|
| Best test accuracy | 25.8% (with detectors) | 13.2% (single layer) |
| Mode collapse | Yes (rotates) | Yes (rotates) |
| Weight settling | Partial (some cooling) | None (all HOT) |
| Architecture | 10 detectors + finalizer | Single 784→10 |

Not a fair comparison — ThermalWeight had the detector+finalizer architecture. The flow single-layer result confirms the problem is not in the weight update mechanism.

## Conclusion

Flow weights are mechanically sound (inertia works, tests pass) but cannot overcome the fundamental issue: **the learning signal in a single-layer 10-class classifier is dominated by negative pressure.** The weight update mechanism (thermal vs flow) is not the bottleneck. The architecture and target formulation need to change first.

## Next Steps

1. **Fix targets**: try symmetric (±30), or one-vs-rest (only update the correct class and the predicted class, not all 10)
2. **Add hidden layer**: 784 → 128 → 10 to extract features before classification
3. **Use flow weights in the detector architecture**: replace ThermalWeight with FlowWeight in the existing 10-detector + finalizer pipeline
