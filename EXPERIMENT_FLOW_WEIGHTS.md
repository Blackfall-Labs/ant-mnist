# Experiment: Flow-Based Signal Weights

## Date: 2026-04-03

## Hypothesis

The current ThermalWeight system (discrete pressure thresholds, hit-based cooling, ±1 magnitude transitions) causes oscillation and mode collapse because:

1. Weight changes are too coarse (±1 per transition)
2. Pressure resets on transition, losing directional context
3. Temperature is decoupled from actual weight stability
4. No inertia — weights can flip direction instantly

A momentum-based "flow weight" system should fix this by:
- Making nudges proportional to error × input (gradient-like precision)
- Accumulating momentum with decay (inertia prevents flip-flopping)
- Deriving temperature from momentum stability (not hit counts)
- Allowing weights to settle naturally instead of threshold-gating

## Design: FlowWeight

```
[value: i16, momentum: i16, volatility: u16]  — 6 bytes per weight

value      = current weight position, signed, full range
momentum   = velocity of change (accumulated nudges × decay)
volatility = EMA of |momentum|, drives derived temperature
```

### Update rule (per step)

```
scaled_nudge = nudge × temp_factor / 256
momentum     = momentum × 0.898 + scaled_nudge
value        = value + momentum / 16
volatility   = volatility × 0.977 + |momentum| × 0.023
temperature  = clamp(volatility, 0, 255)
```

### Key properties

- **Inertia**: reversing direction requires overcoming existing momentum
- **Settling**: momentum decays, so weights naturally find equilibrium
- **Temperature = stability**: volatile weights are HOT, settled weights are COLD
- **No transitions**: weights move continuously, not in ±1 discrete steps
- **No pressure threshold**: every nudge contributes, proportional to error

## Experimental Setup

### Architecture

**Experiment A: Single-layer flow classifier**
- 784 → 10 FlowMatrix (direct pixel-to-class)
- Baseline comparison: how does raw flow learning perform?
- No detectors, no correlation, just flow weights on raw pixels

**Experiment B: Detector + finalizer with flow weights**
- 10 × (1 × 784 FlowMatrix) detectors + 1 × (10 × 7840 FlowMatrix) finalizer
- Same three-phase curriculum as current system
- Direct comparison against ThermalWeight version

### Training Protocol

For each experiment:
1. **Warm-up**: 10 cycles of 200 samples/digit (balanced)
2. **Main training**: 90 cycles of 200 samples/digit (balanced)
3. **Evaluation**: 1000 test samples after every 5 training cycles

Total: 100 cycles, ~20K training samples per digit.

### Metrics to Record (per eval point)

| Metric | What it measures |
|--------|-----------------|
| `test_accuracy` | Overall correct / total on test set |
| `per_class_accuracy` | Accuracy breakdown per digit (0-9) |
| `confusion_matrix` | Full 10×10 matrix |
| `mode_collapse_score` | max(per_class_acc) - mean(per_class_acc). 0 = uniform, high = collapse |
| `weight_magnitude_mean` | Average |value| across all weights |
| `weight_magnitude_std` | Std dev of |value| — are weights differentiating? |
| `momentum_mean` | Average |momentum| — is the system still learning or settled? |
| `temperature_distribution` | Count of weights in HOT/WARM/COOL/COLD bands |
| `nonzero_weights` | How many weights have developed non-zero values |
| `cycle_time_ms` | Wall clock per training cycle |
| `weight_count` | Total weights in the system |
| `weight_bytes` | Total memory for weights (weight_count × 6 bytes) |
| `peak_rss_mb` | Peak resident memory of the process |
| `cpu_util` | Approximate CPU utilization during training |

### Success Criteria

| Criterion | Threshold | Why |
|-----------|-----------|-----|
| Test accuracy > 30% | Above current best (25.8%) | Flow weights learn better than thermal |
| No mode collapse | collapse_score < 0.5 | Multiple detectors contributing |
| Stable accuracy | Accuracy doesn't oscillate > ±5% between evals | Momentum prevents flip-flopping |
| Temperature gradient | All 4 bands populated after 50 cycles | Weights differentiate in stability |
| Monotonic cooling | Mean temperature decreases over training | System converges, doesn't yo-yo |

### Failure Modes to Watch

| Failure | Signal | Meaning |
|---------|--------|---------|
| Momentum explosion | values hit ±32000 | Nudge scale too large, needs dampening |
| Permanent HOT | All weights stay > 192 | Momentum decay too slow, nothing settles |
| Premature COLD | All weights < 64 by cycle 20 | Volatility decay too fast |
| Uniform weights | magnitude_std near 0 | Nudges too small, weights not differentiating |
| Accuracy plateau at 10% | Same as random | Learning signal not reaching weights |

### Constants to Tune

| Constant | Initial value | Range to try | Effect |
|----------|--------------|-------------|--------|
| `MOMENTUM_DECAY` | 230/256 (0.898) | 200-250 | Higher = more inertia, slower direction change |
| `VOLATILITY_DECAY` | 250/256 (0.977) | 240-254 | Higher = slower temperature changes |
| `VALUE_MIN/MAX` | ±500 | ±200, ±1000 | Weight range — too small = can't learn, too large = unstable |
| `momentum / 16` divisor | 16 | 8, 32 | Momentum-to-position coupling — how fast weights move |
| Nudge scale `/1000` | 1000 | 500, 2000 | How much error × input contributes to momentum |
| Target values | +40/-20 | +60/-30, +20/-10 | Asymmetry between positive and negative targets |

## Comparison Baseline

Current ThermalWeight results (100 cycles, three-phase curriculum):
- Best test accuracy: 25.8%
- Typical eval: 7-15%
- Mode collapse: d4 or d1 at 91-100%, all others at 0%
- Oscillation: accuracy swings ±10% between evals
- Thermal distribution: mostly uniform HOT/WARM, minimal differentiation

## Procedure

1. Run Experiment A (single-layer) first — fastest to iterate, proves the weight mechanics
2. Record all metrics to CSV (one row per eval point)
3. If Experiment A shows promise (>15% on single layer), proceed to Experiment B
4. If Experiment A fails, tune constants and re-run before adding complexity
5. Compare Experiment B against ThermalWeight baseline on identical curriculum

## Output Format

Metrics logged to `experiments/flow_v1.csv`:

```csv
experiment,cycle,test_acc,per_class_acc,collapse_score,weight_mag_mean,weight_mag_std,momentum_mean,temp_hot,temp_warm,temp_cool,temp_cold,nonzero,weight_count,weight_bytes,peak_rss_mb,cycle_ms
A,5,0.12,"0.10|0.15|0.08|...",0.42,12.3,8.7,45.2,1200,3000,2800,840,6234,7840,47040,85.2,150
```
