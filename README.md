# ANT-MNIST

**CPU-only MNIST digit classification using Atomic Neural Transistors.**

No gradients. No floats. No GPU. Ternary signal processing with pressure-based mastery learning on integer substrates.

---

## How It Works

### Architecture: Correlation Detectors → Finalizer

```
Input Image (28×28 = 784 pixels)
  ↓ encode
784 ternary signals (Polarity × Magnitude × Multiplier)
  ↓ fan out to 10 detectors
┌──────────────────────────────────────────────────┐
│ Detector 0 (imprinted on "0"): correlate → 784  │
│ Detector 1 (imprinted on "1"): correlate → 784  │
│ ...                                              │
│ Detector 9 (imprinted on "9"): correlate → 784  │
└──────────────────────────────────────────────────┘
  ↓ concatenate
7840 correlation signals
  ↓ finalizer ANT
10 class scores
  ↓ argmax
Predicted digit (0-9)
```

**Detectors** are imprinted — not trained. Each detector absorbs examples of its target digit, building a spatial template. When shown an input image, it produces 784 confidence signals: one per pixel position, where each signal encodes:
- **Polarity**: does this pixel agree (+1) or disagree (-1) with the template?
- **Magnitude**: how strong is the feature at this position?
- **Multiplier**: how consistently was this position active across imprinted samples?

This preserves spatial structure all the way through — the detector doesn't collapse to a single "yes/no." It returns a **field of confidence** showing where and how much the input matches its template.

**The Finalizer** takes all 10 correlation maps (7840 signals total) and learns which spatial patterns of agreement/disagreement reliably indicate each digit. This is where mastery learning happens — pressure-based, integer-only, no gradients.

### How This Differs From Standard ML

| Aspect | Standard ML (CNN/MLP) | ANT-MNIST |
|--------|----------------------|-----------|
| **Representation** | 32-bit floats | Ternary signals (Polarity × Magnitude × Multiplier) |
| **Feature extraction** | Learned via backpropagation | Imprinted from examples (direct pattern absorption) |
| **Classification** | Softmax over learned features | Mastery learning over correlation patterns |
| **Learning rule** | Gradient descent (calculus) | Pressure accumulation → threshold → transition (integer) |
| **Hardware** | GPU (massive float parallelism) | CPU (integer ops only) |
| **Weight updates** | Continuous (any float value) | Discrete (weaken-before-flip, step up/down) |
| **Plasticity control** | Learning rate schedules | Per-weight thermal state (HOT/WARM/COOL/COLD) |
| **Runtime** | Python + PyTorch, compiled graph | Runes scripts, hot-reloadable without recompilation |
| **Initialization** | Random (Xavier, He, etc.) | Imprinting (direct pattern absorption from examples) |

### How This Connects to the Original MNIST (LeCun 1998)

LeCun's original LeNet-5 used:
- Convolutional layers to extract spatial features (edges, strokes)
- Subsampling (pooling) to build translation invariance
- Fully connected layers for classification
- Backpropagation with gradient descent
- Achieved 99.05% on the full test set

Our approach replaces every component:
- **Convolution → Imprinted templates.** Instead of learning edge detectors through thousands of gradient updates, we directly absorb digit patterns from examples. The imprinted template IS the feature detector.
- **Pooling → Signal correlation.** Instead of downsampling to build invariance, the element-wise correlation naturally handles positional variation — a stroke at position [5,12] in the template still partially correlates with a stroke at [5,13] in the input because neighboring pixels have similar magnitudes.
- **Fully connected → Finalizer ANT.** Instead of learned float weights connecting features to classes, the finalizer uses pressure-based mastery learning on ternary signals.
- **Backpropagation → Mastery learning.** Instead of computing gradients through the entire network, only the finalizer learns. The detectors are passive after imprinting.

### Key Findings

**Curriculum sensitivity.** The system responds strongly to training set size per cycle:
- **50 samples/digit** (500 total): Steady improvement. Low variety means pressure accumulates consistently in the right direction. Each pattern repeats enough times to cross mastery threshold.
- **200 samples/digit** (2000 total): Accuracy tanks. Too much handwriting variety per cycle — pressure from one sample gets contradicted by another before threshold is reached. The finalizer can't lock in any pattern.
- **Alternating 50-200-50-200**: Best results. The small set builds foundation (consistent pressure → transitions). The large set stress-tests (exposes weaknesses, corrects errors). The small set re-stabilizes. This pattern produces stable climbing accuracy, reaching 17% from a baseline of 10%.

This mirrors biological learning: focused repetition (drilling) builds skill, varied exposure (generalization) tests it, then focused repetition repairs what broke.

**Imprinting vs random projection.** Random hidden projections (the standard approach in reservoir computing) failed completely on MNIST — the projections were too similar across detectors and produced no discriminative features. Imprinting from actual digit examples produces templates that reflect real spatial structure: the "1" template has a vertical green stripe, the "0" has a green loop, the "7" has a horizontal bar. These are meaningful feature detectors, not noise.

**The detector-as-correlator model.** Treating detectors as spatial correlation reporters (returning 784 signals) rather than binary classifiers (returning 1 scalar) preserves information that the finalizer needs. The finalizer can learn "detector 5 fires strongly in the top-left but weakly in the bottom means this is a 5, not a 6" — something a single confidence scalar couldn't express.

**Output weight initialization matters.** Mastery learning cannot transition weights from zero — pressure needs something to push against. The original approach (all-zero output weights) produced permanent mode collapse because every detector returned the same value and argmax always picked the last tied index.

---

## Running

### Control Room (Dashboard)

```bash
cargo run --release --bin ant-mnist-viz [data_dir] [port]
```

Opens a web dashboard at `http://localhost:8090` with:
- **Per-detector panels**: template heatmap (green=excitatory, red=inhibitory), correct/wrong examples, confidence sparkline
- **Operator controls**: Imprint All, Train Cycle (50/digit), Train Cycle (200/digit), Evaluate, Freeze All
- **Per-detector controls**: Imprint, Train, Freeze, Thaw, Flush
- **Confusion matrix**: 10×10 grid showing classification distribution
- **Accuracy history**: test accuracy over cycles
- **Action log**: every operation logged with results

### Operator Workflow

1. **Imprint All (50)** — detectors absorb 50 examples each. Heatmaps populate.
2. **Evaluate (1000)** — see baseline accuracy and confusion matrix.
3. **Train Cycle (50/digit)** — train finalizer on small balanced set. Repeat 5×.
4. **Train Cycle (200/digit)** — stress test with more variety. Repeat 2×.
5. **Train Cycle (50/digit)** — re-stabilize. Repeat 5×.
6. **Evaluate (1000)** — check progress.
7. Repeat 3-6, watching the confusion matrix and accuracy curve.

### MNIST Data

Place the IDX files in `data/`:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

---

## Architecture

```
ant-mnist/
  runes/
    digit_detector.rune  — spatial correlation detector (imprint + correlate)
    finalizer.rune       — 7840→10 classification (mastery learning)
  src/
    main.rs              — Runes-first runtime (loads .rune as the program)
    viz.rs               — Control room dashboard
    viz_specialist.html  — Dashboard frontend
    mnist.rs             — IDX file parser
    mnist_module.rs      — MNIST-specific Runes verbs
```

### Dependencies

- **atomic-neural-transistors** — ANT runtime, mastery learning, thermal weights
- **runes** — Hot-reloadable behavioral scripting (meta/main/hook/def)
- **ternary-signal** — Signal type (Polarity × Magnitude × Multiplier)

---

## Status

- Architecture proven: correlation detectors + finalizer produces above-random classification
- Mastery learning responds to curriculum tuning (50/200 alternation)
- Dashboard provides operator control over training process
- Accuracy: 17% best (from 10% baseline) — early stage, needs continued tuning
- Next: React/WebSocket dashboard for real-time observation, continued mastery tuning, expanded curriculum strategies

---

*Blackfall Labs — Neuromorphic systems. Not machine learning.*
