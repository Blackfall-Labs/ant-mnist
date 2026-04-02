# ML Operations Guide — What They Actually Do

This is a reference for understanding what the mathematical operations borrowed from ML actually compute, why they exist, and how they map to what ANTs do with ternary signals. No jargon assumed.

---

## 1. matmul (Matrix Multiplication)

**What it does:** Takes every input signal and combines it with every synaptic strength to produce an output signal. It's a weighted sum — each output neuron collects contributions from every input, scaled by its connection strength to that input.

**Concretely:** If you have 784 inputs and 300 outputs, there are 300×784 = 235,200 synaptic strengths. Each of the 300 output neurons does:

```
output[i] = input[0] × strength[i][0]
           + input[1] × strength[i][1]
           + ...
           + input[783] × strength[i][783]
```

**Why it matters:** This is _the_ core operation. Every region in the network is a matmul — it's how signals propagate through connections. The synaptic strength matrix IS the learned knowledge.

**In ANTs:** `ThermalWeightMatrix::matmul()` or the `matmul` Runes verb. Uses integer arithmetic on PackedSignal values. Each "weight" is a ternary signal (Polarity × Magnitude × Multitude).

**Analogy:** Think of it like 300 microphones each listening to 784 speakers. Each microphone has its own volume knob for each speaker. The matmul computes what each microphone hears.

---

## 2. ReLU (Rectified Linear Unit)

**What it does:** If a signal is positive, keep it. If it's negative or zero, make it zero.

```
relu(x) = max(0, x)

relu(42)  → 42
relu(-17) → 0
relu(0)   → 0
```

That's it. That's the whole thing.

**Why it matters:** Without ReLU (or something like it), stacking multiple matmuls is pointless — two matmuls in a row collapse into a single matmul. The math is: `(A × B) × input = C × input` where `C = A × B`. You'd get the same result with one region as with ten.

ReLU breaks this linearity. By zeroing out negatives, it creates a _selection effect_ — different inputs activate different subsets of neurons. This is what lets the network develop different internal representations for different inputs (e.g., the digit "3" activates a different set of neurons than "7").

**In ANTs:** `relu_packed()` or the `relu` Runes verb. Checks `PackedSignal::current()`, zeros out anything ≤ 0.

**What it's like biologically:** It's a crude version of a firing threshold. A neuron that hasn't accumulated enough excitatory input simply doesn't fire. The real biological version has resting potential, leak, refractory periods — ReLU is the simplest possible version of "silent unless excited enough."

---

## 3. Sigmoid

**What it does:** Squashes any value into the range (0, 1). Large positive → close to 1. Large negative → close to 0. Zero → exactly 0.5.

```
sigmoid(x) = 1 / (1 + e^(-x))

sigmoid(10)  → 0.9999...  (almost 1)
sigmoid(-10) → 0.0000...  (almost 0)
sigmoid(0)   → 0.5
sigmoid(2)   → 0.88
sigmoid(-2)  → 0.12
```

The curve is an S-shape (sigmoid literally means "S-shaped").

**Why it matters:** It acts as a _gate_ — a soft yes/no decision. In the classifier rune, sigmoid is used on the gate matrix: it decides "how much of this update should I let through?" Values near 1 = let it through. Values near 0 = block it.

**In ANTs:** The `sigmoid` Runes verb approximates this in integer space. Since PackedSignal is integer-based, it maps the current value to a 0-255 range using a lookup or piecewise approximation.

**In the classifier rune:**

```
gate = matmul(h, w_gate, 24, 24)
gate = sigmoid(gate)
update = mul(update, gate)    ← gate controls how much update gets through
```

This is a gated recurrent pattern — the gate learns which dimensions of the hidden state should be updated and which should be preserved.

---

## 4. Softmax

**What it does:** Takes a vector of raw scores and converts them into probabilities that sum to 1. The highest score gets the highest probability. It amplifies differences — a small gap in raw scores becomes a large gap in probabilities.

```
softmax([3, 1, 0.5]) → [0.80, 0.11, 0.07]
softmax([10, 10, 10]) → [0.33, 0.33, 0.33]  (equal inputs → equal probabilities)
softmax([100, 1, 1])  → [1.00, 0.00, 0.00]  (dominant input takes all)
```

**The math:**

```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```

`e^x` is exponential — it grows extremely fast. So even a small difference in raw scores gets amplified exponentially.

**Why it matters for classification:** When you have 10 output neurons (one per digit), their raw outputs might be `[42, 38, -5, 12, ...]`. Softmax turns this into `[0.73, 0.24, 0.00, 0.01, ...]` — a probability distribution. "73% confident it's digit 0, 24% it's digit 1..."

**In ANTs:** The `softmax` Runes verb. In the current ant-mnist raw Rust code, softmax is NOT used — instead it just takes argmax (highest value wins). This is simpler but loses the probability interpretation.

**Do we need it?** For pure classification (just pick the winner), argmax is sufficient. Softmax matters when you need to know _how confident_ the system is, or when you're computing a more nuanced error signal during learning. In ant-mnist, we might benefit from it for better error signals.

---

## 5. Argmax

**What it does:** Returns the index of the largest value.

```
argmax([3, 7, 2, 5]) → 1  (because 7 is at index 1)
argmax([0, 0, 9, 0]) → 2
```

**In ANTs:** This is what ant-mnist does for classification — find the output neuron with the highest current, and that's the predicted digit. The `argmax` Runes verb.

---

## 6. One-Hot Target

**What it does:** Represents a class label as a vector where one position is "on" and all others are "off."

```
For digit "3" with 10 classes:
one_hot = [off, off, off, ON, off, off, off, off, off, off]
```

**In ant-mnist currently:**

```rust
target[correct_label] = packed_from_current(60);    // ON = excitatory
everything else       = packed_from_current(-40);    // OFF = inhibitory
```

**Why it matters:** This is how you tell the network what the right answer was. "Neuron 3 should have fired strongly. All others should have been inhibited." The error is then: `target - actual_output` for each output neuron.

---

## 7. Error / Target Derivation for Hidden Regions

This is the hard one, and it's where ant-mnist is currently broken.

**The problem:** You know what the output SHOULD have been (the one-hot target). You can compute the error at the output: `target - actual`. But what should the _hidden_ neurons have done? Nobody told them a target. They're internal.

**In standard ML (backpropagation):** You use calculus to compute exactly how much each hidden neuron contributed to the output error, and push an error signal backwards through the network. This requires continuous (float) math and is the thing Blackfall explicitly doesn't do.

**What ant-mnist does now (the broken version):**

```rust
drive += (err.signum() * w.signum()) as i32;
```

This asks: "Was the error positive or negative? Was the connection positive or negative?" and derives a ±1 nudge. It works for toy problems but loses all magnitude information — it can't tell "slightly wrong" from "catastrophically wrong."

**What the thermal_classifier example does (the better version):**

```rust
let scaled_error = errors[c] / 4;
nudge += scaled_error * w_sign;
```

This keeps the error _magnitude_ and only reduces the synaptic strength to its sign. This means large errors produce large nudges and small errors produce small nudges. Much better signal.

**What we should try:** The thermal_classifier approach, adapted for MNIST's 10 classes. And potentially: only propagate error from the correct class and the predicted class (2 contributors), not all 10. Most output neurons had nothing useful to say about what the hidden neurons should do.

---

## 8. Gated Recurrence (the classifier.rune pattern)

The classifier rune has a pattern that's NOT in ant-mnist but is worth understanding:

```rune
update = matmul(h, w_rec, 24, 24)    # "What should change?"
update = relu(update)
gate = matmul(h, w_gate, 24, 24)     # "Should it change?"
gate = sigmoid(gate)
update = mul(update, gate)            # Gate controls the update
h = add(h, update)                    # Apply gated update
h = shift(h, 1)                       # Right-shift (divide by 2)
```

This is a simplified GRU (Gated Recurrent Unit). The idea:

- **w_rec** proposes an update based on current state
- **w_gate** decides how much of that update to allow (0 = block, 1 = full update)
- The update is added to the current state (residual connection)
- The shift prevents values from growing unboundedly

This pattern lets the network iterate on its internal representation — look at the same input multiple times and refine its understanding. The classifier rune does 3 iterations.

**For MNIST:** We probably don't need recurrence for static images. A feedforward cascade (matmul → relu → matmul → relu → matmul → argmax) should suffice. Recurrence is more useful for sequential data like audio (hush).

---

## 9. Shift

**What it does:** Arithmetic right-shift. `shift(x, 1)` divides every value by 2 (rounding toward zero).

```
shift(100, 1) → 50
shift(7, 1)   → 3
shift(x, 2)   → x / 4
```

**Why:** Prevents signal values from growing without bound as you add and multiply. After adding a recurrent update to the hidden state, shift keeps values in a manageable range. It's a cheap normalization.

**In ANTs:** The `shift` Runes verb. For MNIST feedforward, we probably don't need this.

---

## 10. mul, add, sub

Element-wise arithmetic on signal vectors.

```
mul([a, b, c], [x, y, z]) → [a×x, b×y, c×z]
add([a, b, c], [x, y, z]) → [a+x, b+y, c+z]
sub([a, b, c], [x, y, z]) → [a-x, b-y, c-z]
```

**In the gate pattern:** `mul(update, gate)` multiplies each update dimension by its gate value. Gate of 0 = no update. Gate of 1 = full update.

---

## 11. Normalize

**What it does:** Scales a vector so its values span a consistent range. Prevents one region's output from dominating the next region's input just because its raw values happen to be larger.

**Why it matters for ANTs:** Integer arithmetic can overflow or saturate. If matmul produces values in the millions (784 inputs × magnitude 60 × strength 40 = up to 1.8M per neuron), the next matmul will produce even larger values. Normalization keeps things in bounds.

**In ant-mnist currently:** There's no explicit normalization. The current code clamps outputs to [-127, 127] in some places, but this is aggressive — it destroys information. Shift-based normalization (divide by a power of 2) is gentler.

---

## 12. The MNIST Classification Pipeline

Putting it all together, here's what a feedforward MNIST classifier does:

```
Image (28×28 = 784 pixels)
  ↓ encode
784 PackedSignals (ternary: excitatory/silent/inhibitory)
  ↓ matmul(input, w1)     ← 784 → 300
300 raw signals
  ↓ relu
300 signals (negatives zeroed)
  ↓ matmul(h1, w2)        ← 300 → 100
100 raw signals
  ↓ relu
100 signals (negatives zeroed)
  ↓ matmul(h2, w3)        ← 100 → 10
10 output signals (one per digit)
  ↓ argmax
Predicted digit (0-9)
```

The three synaptic strength matrices (w1, w2, w3) are where all the knowledge lives. Mastery learning adjusts them based on whether the prediction was correct.

---

## 13. Why 784 → 300 → 100 → 10?

**784:** MNIST images are 28×28 = 784 pixels. Fixed by the data.

**300, 100:** These are arbitrary choices for internal capacity. More neurons = more capacity to represent different patterns, but also more synaptic strengths to learn. The thermal_classifier example uses much smaller numbers (32 → 24 → 4) because its problem is much simpler (4 classes, 32-dim input).

**10:** MNIST has 10 digit classes (0-9). Fixed by the problem.

**Should we change these?** The handoff suggests trying 784→500→200→10 (more capacity) or 784→100→10 (simpler, faster to learn). Given the current learning problems, simpler might be better to debug with.

---

## 14. What Makes ANT Classification Different from ML

| Aspect                    | Standard ML                       | ANTs                                                                           |
| ------------------------- | --------------------------------- | ------------------------------------------------------------------------------ |
| **Values**                | 32-bit floats                     | Integer PackedSignals (Polarity × Magnitude × Multitude)                       |
| **Learning**              | Gradient descent (calculus-based) | Mastery learning (pressure accumulation → discrete transitions)                |
| **Synaptic strengths**    | Continuous floats, any value      | Ternary signals (Polarity × Magnitude × Multitude), pressure-gated transitions |
| **Temperature**           | N/A                               | Thermal state gates plasticity (HOT=plastic, COLD=frozen)                      |
| **Hardware**              | GPU (massive float parallelism)   | CPU (integer ops, no GPU needed)                                               |
| **Runtime**               | Python/PyTorch, compiled graph    | Runes scripts, hot-reloadable                                                  |
| **When to stop learning** | Loss converges                    | Mastery: neurons demonstrate consistent correct behavior, then cool/freeze     |

The operations (matmul, relu, argmax) are the same math. The difference is _what's being multiplied_ (ternary signals — Polarity × Magnitude × Multitude — not floats), _how learning happens_ (pressure not gradients), and _how plasticity is controlled_ (thermal state not learning rate schedules). DeltaTime may factor into the signal formula as well but hasn't been nailed down yet.

---

## 15. Glossary of Terms Used Above

| Term                    | Meaning                                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **vector**              | An ordered list of numbers. `[3, -1, 0, 7]` is a 4-element vector.                                                                              |
| **matrix**              | A 2D grid of numbers. A synaptic strength matrix with 300 rows and 784 columns has 235,200 values.                                              |
| **element-wise**        | Operation applied to each position independently. `add([1,2,3], [4,5,6]) = [5,7,9]`.                                                            |
| **linear**              | A transformation where doubling the input doubles the output. `matmul` is linear.                                                               |
| **nonlinearity**        | Something that breaks linearity. ReLU is a nonlinearity (doubling a negative input doesn't double the output — it stays zero).                  |
| **saturation**          | When values hit a ceiling/floor and stop carrying useful information. All outputs being 127 or -127 = saturated.                                |
| **convergence**         | When the system reaches a stable state where accuracy stops improving.                                                                          |
| **mode collapse**       | When the system always predicts the same answer regardless of input.                                                                            |
| **GRU**                 | Gated Recurrent Unit — the pattern from the classifier rune. A way to iteratively refine an internal representation.                            |
| **residual connection** | Adding the input to the output (`h = add(h, update)`), so the network only needs to learn _what to change_, not the entire output from scratch. |
| **e (Euler's number)**  | ≈ 2.71828. The base of natural exponentials. `e^x` grows faster than any polynomial. Used in sigmoid and softmax.                               |
