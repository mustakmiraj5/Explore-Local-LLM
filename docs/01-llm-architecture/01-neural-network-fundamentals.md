# 1.1 Neural Network Fundamentals

> Topic 1: LLM Architecture · Lesson 1 of 5 · Status: 🟡 in progress

---

## 1. Introduction

**What is it?**
A neural network is a **mathematical function with millions or billions of adjustable numbers inside it**. It takes numbers in and gives numbers out. Training means finding values for those internal numbers so the outputs are useful.

**Why does it exist?**
Some problems can't be solved by writing rules by hand. Nobody can write `if/else` logic for "what word comes next in this sentence?" There are too many cases. A neural network turns the question around: instead of *writing* the function, you *search* for one by adjusting numbers until it fits a huge pile of examples.

**Why does it matter for LLM engineering?**
Qwen 3.5 4B is a neural network. When you call it through Ollama, your machine does nothing but **multiply your input by billions of stored numbers**. Speed, memory use, quantization, context limits and hardware requirements all come from that fact. If you understand the basic unit, the rest of the syllabus follows from it.

```
   Traditional software            Neural network
   ────────────────────            ──────────────
   input ──► [rules you wrote] ──► output
   input ──► [numbers learned from data] ──► output
```

---

## 2. Fundamentals

### 2.1 The neuron: one weighted vote

A single artificial **neuron** does three things:

1. Multiplies each input by a **weight** (how much that input matters)
2. Adds them all up, plus a **bias** (a baseline offset)
3. Passes the result through an **activation function** (a simple non-linear "bend")

```
  x1 ──(×w1)──┐
              │
  x2 ──(×w2)──┼──► Σ + b ──► activation ──► output
              │    (z)        f(z)
  x3 ──(×w3)──┘
```

As a formula: `output = f(w1·x1 + w2·x2 + w3·x3 + b)`

**Analogy:** You're choosing whether to deploy on a Friday. The inputs are: tests pass (1/0), it's 5 pm (1/0), and the change is risky (1/0). You give each a weight: tests passing counts **+3**, 5 pm counts **−2**, risky counts **−4**. Add them up with your baseline mood (bias). If the total is positive, deploy. That's a neuron. The weights encode *judgment*.

### 2.2 Terminology

| Term | Meaning |
|---|---|
| **Weight** | A multiplier on one input. It's learned. |
| **Bias** | A constant added after the weighted sum. It's learned. |
| **Parameters** | All weights and biases together. "4B" means about 4 billion of these. |
| **Activation function** | A fixed non-linear function applied to the sum, like ReLU: `max(0, z)` |
| **Layer** | Many neurons that read the same inputs side by side |
| **Forward pass** | Running inputs through all layers to get an output |
| **Vector** | A list of numbers, like `[0.2, -1.3, 0.7]`. It's the "data type" passed between layers. |

### 2.3 Layers: neurons in parallel, then stacked

One neuron can't do much. Put many side by side and you get a **layer**. Each neuron in the layer reads the same input vector but has its own weights, so each one detects something different. Stack layers so that the output of one becomes the input of the next:

```
 input vector     hidden layer      hidden layer     output
 (3 numbers)      (4 neurons)       (4 neurons)      (2 numbers)

    x1 ─────┬──►  ○ ───────┬──►  ○ ─────┬──►  ○  y1
    x2 ─────┼──►  ○ ───────┼──►  ○ ─────┼──►  ○  y2
    x3 ─────┴──►  ○ ───────┴──►  ○ ─────┘
                  ○                ○
           (every arrow = one weight)
```

**Why stack layers?** Each layer builds on the one before it. In image models, early layers detect edges, middle layers detect shapes, and late layers detect "cat". In LLMs, it's similar but harder to see: early layers handle surface patterns, later layers handle meaning and the next-word decision.

### 2.4 Why the activation function is essential

This is the most important "why" in this lesson.

A weighted sum is a **linear** operation: it can only draw straight lines (or flat planes). If you stack linear layers without activations, the result is still one straight line. The layers **collapse** into a single layer, and extra depth adds nothing.

Activation functions add a **bend**. With enough bends, a network can approximate almost any shape of function. That's what makes neural networks powerful. Formally, this is called the *universal approximation* property.

```
  ReLU(z) = max(0, z)

  output
    │        ╱
    │      ╱
    │    ╱
  ──┼──── ─ ─ ─ ── z
    0
    flat for negatives, straight for positives: one simple bend
```

You'll test this collapse yourself in Section 4.

### 2.5 Where do the weights come from? (training, in one paragraph)

At the start, the weights are random. The network predicts something, and we measure **how wrong** it was with a number called the **loss**. Calculus (the **gradient**) tells us, for each weight, *which direction to nudge it so the loss goes down*. Nudge every weight a tiny bit and repeat millions of times. That loop is **gradient descent**. The weights end up holding whatever patterns reduce error on the training data.

```
 random weights ─► predict ─► measure error ─► nudge weights ─┐
        ▲                                                     │
        └─────────────────── repeat ──────────────────────────┘
```

> Training happens **once**, on huge GPU clusters, before you download the model.
> When you run Qwen in Ollama, the weights are **frozen**. Your machine only does forward passes.
> (Training vs inference gets its own lesson in Topic 4.)

---

## 3. Under the Hood

### 3.1 A layer is a matrix multiplication

Writing neurons one at a time is fine for learning, but real systems compute a whole layer at once. If a layer has `N` neurons and each reads `D` inputs:

- Store all the weights in a matrix **W** with shape `[N × D]`. Each **row** is one neuron's weights.
- Store the biases in a vector **b** with `N` entries.
- Then the whole layer is a single operation:

```
  output = f( W · x + b )

  W: [N × D]    x: [D]    b: [N]    output: [N]
```

Worked example with 2 neurons and 3 inputs:

```
      W              x         b          z
 ┌ 0.5  -1   2 ┐   ┌ 1 ┐    ┌ 0.1 ┐   ┌ 0.5·1 + (-1)·2 + 2·3 + 0.1 = 4.6 ┐
 └ 1     0  -1 ┘ · │ 2 │  + └ -0.5┘ = └ 1·1  +  0·2  + (-1)·3 - 0.5 = -2.5┘
                   └ 3 ┘
 ReLU → [4.6, 0.0]
```

**Why this matters for engineering:** an LLM is mostly **huge matrix multiplications**. GPUs, Apple's Metal GPU and the Neural Engine are all built to do many multiply-adds in parallel. That's why LLMs run on GPUs, and why your M4's unified memory and GPU matter more than its CPU clock speed.

### 3.2 The real thing: a layer inside your Qwen model

Your local model's weights aren't abstract. You can list them. One of Qwen's internal layers (part of its feed-forward block, which you'll study in 1.3) looks like this:

```
language_model.model.layers.0.mlp.up_proj.weight     [9216 2560]
language_model.model.layers.0.mlp.down_proj.weight   [2560 9216]
```

How to read it:
- `up_proj` is a weight matrix with **9,216 neurons, each with 2,560 weights**. It takes a 2,560-number vector and expands it to 9,216 numbers.
- `down_proj` shrinks it back to 2,560.
- `layers.0` means this is the first of **32** stacked layers (`block_count = 32`).
- 2,560 is the model's **hidden size** (`embedding length`): the length of the vector that flows through the whole network.

That's exactly the `W · x` from 3.1, just much bigger. There's also no `.bias` tensor here. Many modern LLMs drop biases in these layers because they add little at this scale.

### 3.3 How this connects to the rest of an LLM

```
 "The cat sat"  ──► tokenizer ──► token IDs ──► vectors (2560 numbers each)
                                                  │
                          ┌───────────────────────┘
                          ▼
                 ┌─────────────────┐
                 │ layer 1         │  ← attention + neural network layers
                 │ layer 2         │     (matrix multiplications + activations)
                 │ ...             │
                 │ layer 32        │
                 └────────┬────────┘
                          ▼
              scores for every possible next token ──► pick one ──► "on"
```

Everything between "vectors" and "scores" is built from the neuron and layer you just learned. **Attention** (lesson 1.4) is the one new ingredient. It lets the vectors for different words exchange information with each other.

---

## 4. Practical Implementation

### 4.1 A neuron and a tiny network in plain Python

No libraries. Standard library only, so nothing is hidden. Save this as `nn_basics.py` and run it with `python3 nn_basics.py`.

The task is **XOR**: output 1 if exactly one input is 1. It's the classic problem a single straight line **can't** solve.

```python
def relu(z):
    return max(0.0, z)

def neuron(inputs, weights, bias, activation=relu):
    z = sum(x * w for x, w in zip(inputs, weights)) + bias   # weighted sum
    return activation(z)

def tiny_network(x1, x2, activation=relu):
    # Hidden layer: 2 neurons, each looks at both inputs
    h1 = neuron([x1, x2], weights=[1.0, 1.0], bias=0.0,  activation=activation)
    h2 = neuron([x1, x2], weights=[1.0, 1.0], bias=-1.0, activation=activation)
    # Output layer: 1 neuron, looks at the hidden layer (no activation)
    return neuron([h1, h2], weights=[1.0, -2.0], bias=0.0, activation=lambda z: z)

print("x1 x2 | XOR | with ReLU | without ReLU")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    print(f" {x1}  {x2} |  {x1 ^ x2}  |   {tiny_network(x1, x2):5.1f}   |   "
          f"{tiny_network(x1, x2, activation=lambda z: z):5.1f}")
```

Output:

```
x1 x2 | XOR | with ReLU | without ReLU
 0  0 |  0  |     0.0   |     2.0
 0  1 |  1  |     1.0   |     1.0
 1  0 |  1  |     1.0   |     1.0
 1  1 |  0  |     0.0   |     0.0
```

**What happened:**
- `h1` fires when **at least one** input is on. `h2` fires only when **both** are on, because its bias of −1 is a threshold.
- The output computes `h1 − 2·h2`, which means "at least one, but not both". That's XOR.
- Without ReLU, the math simplifies to `2 − x1 − x2`, a flat plane. It gets `(0,0)` wrong, and **no choice of weights** can fix that. That's the collapse from 2.4, shown in real output.

I chose these weights by hand. A real network *learns* them. Here's that process.

### 4.2 Watching weights get learned (gradient descent)

```python
# Data generated by a hidden rule: y = 3x + 2. The model doesn't know the rule.
data = [(x, 3 * x + 2) for x in [-2, -1, 0, 1, 2, 3]]

w, b = 0.0, 0.0          # parameters start knowing nothing
lr = 0.05                # learning rate: how big each nudge is

for step in range(201):
    grad_w = grad_b = loss = 0.0
    for x, y_true in data:
        y_pred = w * x + b                 # forward pass
        error = y_pred - y_true
        loss += error ** 2                 # squared error
        grad_w += 2 * error * x            # d(loss)/dw
        grad_b += 2 * error                # d(loss)/db
    n = len(data)
    w -= lr * grad_w / n                   # nudge weights against the gradient
    b -= lr * grad_b / n
    if step % 40 == 0:
        print(f"step {step:3d}  loss={loss / n:8.4f}  w={w:.3f}  b={b:.3f}")
```

Output:

```
step   0  loss= 38.5000  w=1.050  b=0.350
step  40  loss=  0.0009  w=3.006  b=1.972
step  80  loss=  0.0000  w=3.000  b=1.999
...
```

The model found `w=3, b=2` **only from examples**. It never saw the rule. Qwen's training is the same loop, scaled to about 4.5 billion parameters and trillions of tokens of text. The "rule" it recovers is "what text tends to come next".

### 4.3 Look inside your own model with Ollama

```bash
# High-level facts
ollama show qwen3.5:4b

# Every weight tensor, with its shape and storage format
ollama show qwen3.5:4b --verbose | grep "layers.0.mlp"
ollama show qwen3.5:4b --verbose | grep -E "parameter_count|block_count|embedding_length"
```

On your machine this reports:

```
general.parameter_count     4.538986496e+09     ← ~4.54 billion learned numbers
qwen3_5.block_count         32                  ← 32 stacked layers
qwen3_5.embedding_length    2560                ← width of the vector between layers
```

**What happens internally when you chat:** Ollama loads those ~4.5 billion numbers into unified memory. For **every token it generates**, it runs the full forward pass, multiplying a 2,560-wide vector through all 32 layers' weight matrices. Nothing is "looked up" or "searched". It's arithmetic all the way through.

---

## 5. Real-World Application

| Neural network fact | Production consequence |
|---|---|
| A model is just weights | A "model file" (like GGUF) is mostly big arrays of numbers. Deploying a model means shipping gigabytes of weights. |
| Inference = matrix multiplications | You need GPUs or accelerators. CPU-only inference works but is slow. |
| Each token needs (roughly) all weights read from memory | **Memory bandwidth**, not raw compute, often limits tokens per second. More on this in Topic 4. |
| Parameter count × bytes per parameter ≈ memory | 4.5B params at 16-bit is about 9 GB, which is tight on your 16 GB Mac. Quantization (Topic 5) shrinks this. Your copy is 4.0 GB. |
| Weights are frozen at inference | Models don't learn from user chats. "Memory" in apps comes from your backend (databases, RAG, prompt history), not the model. |

As a backend engineer, it helps to think of a model as a **stateless, pure function with a very large constant table**. Same input plus deterministic settings gives the same output. All "state" in an LLM product (conversations, user memory, documents) lives in *your* system and gets fed back in as input.

---

## 6. Common Misconceptions

1. **"Neural networks work like the brain."** They were loosely *inspired* by neurons. An artificial neuron is a weighted sum plus `max(0, z)`. Don't reason about LLMs through brain analogies.
2. **"The model stores its training text like a database."** It stores **weights**: compressed statistical patterns. It can sometimes reproduce memorized text, but there's no index, no rows and no lookup. That's why it can confidently produce things that were never in its data (hallucination).
3. **"The model learns while I chat with it."** No. Inference never changes the weights. Anything that looks like learning in a conversation comes from earlier messages being re-sent as input.
4. **"Activation functions are a minor detail."** Without them, any number of layers collapses into one linear function. You saw that in the XOR test.
5. **"Parameters" vs "neurons" vs "layers".** These aren't interchangeable. A layer contains neurons, and each neuron has many parameters (its weights). `up_proj` alone is 9,216 neurons × 2,560 weights ≈ 23.6M parameters.
6. **"More parameters always means a better model."** It means more *capacity*. Quality also depends on data, training and architecture. A well-trained 4B model can beat a poorly trained 13B one.

---

## 7. Knowledge Assessment

Answer in your own words. Short answers are fine. I'll evaluate them and correct any gaps before we move on to **1.2 Parameters and Weights**.

1. **Conceptual:** In plain words, what three operations does a single neuron perform, and which parts of it are learned?
Answer - Weight, Bias, Activation Function. The weights and bias are learned. Activation function is fixed non-linear function applied to the sum.
2. **Conceptual:** Why would a 32-layer network with **no** activation functions be no more powerful than a 1-layer network?
Answer - Because without activation functions, multiple layers produce results that can be represented as a single linear transformation. The layers collapse into a single layer and extra depth adds nothing.
3. **Practical:** A Qwen tensor has shape `[9216 2560]`. What does each number in the shape mean, how many parameters does that matrix hold, and what's the length of its input and output vectors?
Answer - The shape `[9216 2560]` means that the tensor has 9216 rows and 2560 columns. Each row corresponds to a neuron, and each column corresponds to an input feature. The total number of parameters in this matrix is 9216 * 2560 = 23,592,960 parameters. The length of the input vector is 2560, and the length of the output vector is 9216.
4. **Engineering:** You chat with Qwen for an hour and correct it several times. Next day, in a fresh session, it repeats the same mistake. Explain why, in terms of weights.
5. **Reasoning:** In the gradient descent example, what would you expect to happen if `lr` were set to `1.0` instead of `0.05`? (Guess first, then try it.)

---

## 8. Practical Assignment

**Count where Qwen's parameters live.**

1. Run:
   ```bash
   ollama show qwen3.5:4b --verbose | grep "layers.0.mlp"
   ```
2. From the three `mlp` tensor shapes (`gate_proj`, `up_proj`, `down_proj`), compute by hand or in Python:
   - the number of parameters in **one** layer's MLP
   - the total across all **32** layers
   - what **percentage** of the model's 4,538,986,496 parameters that is
3. Write a short Python script that parses the `--verbose` output with `subprocess` and **sums the parameters of every tensor** whose name contains `layers.`. Compare it with `parameter_count`, and explain any difference. (Hint: some tensors aren't inside `layers.`.)

Bring back your numbers and your answers to Section 7.

---

### Progress

- [x] Lesson delivered
- [ ] Assessment answered
- [ ] Assignment completed
- [ ] Ready for 1.2
