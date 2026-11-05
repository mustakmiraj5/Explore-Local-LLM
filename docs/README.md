# LLM Fundamentals — Phase 1

Status legend: ✅ done · 🟡 in progress · ⬜ not started


## 1. Llm Architecture

- 🟡 [1.1 Neural Network Fundamentals](01-llm-architecture/01-neural-network-fundamentals.md)
- ⬜ [1.2 Parameters and Weights](01-llm-architecture/02-parameters-and-weights.md)
- ⬜ [1.3 Transformer Architecture](01-llm-architecture/03-transformer-architecture.md)
- ⬜ [1.4 Attention Mechanism](01-llm-architecture/04-attention-mechanism.md)
- ⬜ [1.5 How the Components Work Together](01-llm-architecture/05-how-the-components-work-together.md)

## 2. Tokenization

- ⬜ [2.1 What Tokens Are](02-tokenization/01-what-tokens-are.md)
- ⬜ [2.2 How Text Becomes Tokens](02-tokenization/02-how-text-becomes-tokens.md)
- ⬜ [2.3 Tokenizers and Vocabulary](02-tokenization/03-tokenizers-and-vocabulary.md)
- ⬜ [2.4 Token IDs](02-tokenization/04-token-ids.md)
- ⬜ [2.5 Tokenization Strategies](02-tokenization/05-tokenization-strategies.md)
- ⬜ [2.6 Tokenization, Cost and Context Windows](02-tokenization/06-tokenization-cost-and-context-windows.md)

## 3. Token Generation

- ⬜ [3.1 Next-Token Prediction](03-token-generation/01-next-token-prediction.md)
- ⬜ [3.2 Logits and Probability Distributions](03-token-generation/02-logits-and-probability-distributions.md)
- ⬜ [3.3 Sampling](03-token-generation/03-sampling.md)
- ⬜ [3.4 Greedy Decoding](03-token-generation/04-greedy-decoding.md)
- ⬜ [3.5 Temperature and Top-p](03-token-generation/05-temperature-and-top-p.md)
- ⬜ [3.6 Autoregressive Generation](03-token-generation/06-autoregressive-generation.md)

## 4. Inference

- ⬜ [4.1 Training vs Inference](04-inference/01-training-vs-inference.md)
- ⬜ [4.2 How Trained Weights Are Used](04-inference/02-how-trained-weights-are-used.md)
- ⬜ [4.3 Forward Propagation](04-inference/03-forward-propagation.md)
- ⬜ [4.4 Prefill and Decode Stages](04-inference/04-prefill-and-decode-stages.md)
- ⬜ [4.5 KV Cache](04-inference/05-kv-cache.md)
- ⬜ [4.6 CPU, GPU and Memory Involvement](04-inference/06-cpu-gpu-and-memory-involvement.md)
- ⬜ [4.7 What Happens When Ollama Receives a Prompt](04-inference/07-what-happens-when-ollama-receives-a-prompt.md)

## 5. Quantization

- ⬜ [5.1 Floating-Point Representations](05-quantization/01-floating-point-representations.md)
- ⬜ [5.2 FP32, FP16, BF16](05-quantization/02-fp32-fp16-bf16.md)
- ⬜ [5.3 INT8 and INT4](05-quantization/03-int8-and-int4.md)
- ⬜ [5.4 Quantization Techniques](05-quantization/04-quantization-techniques.md)
- ⬜ [5.5 Memory and Performance Trade-offs](05-quantization/05-memory-and-performance-trade-offs.md)
- ⬜ [5.6 GGUF and Quantized Models](05-quantization/06-gguf-and-quantized-models.md)

## 6. Context Window

- ⬜ [6.1 What Context Means](06-context-window/01-what-context-means.md)
- ⬜ [6.2 Context Window Limitations](06-context-window/02-context-window-limitations.md)
- ⬜ [6.3 Input and Output Tokens](06-context-window/03-input-and-output-tokens.md)
- ⬜ [6.4 Context Consumption](06-context-window/04-context-consumption.md)
- ⬜ [6.5 KV Cache and Memory Requirements](06-context-window/05-kv-cache-and-memory-requirements.md)
- ⬜ [6.6 Long-Context Challenges](06-context-window/06-long-context-challenges.md)

## 7. Ollama Internals

- ⬜ [7.1 What Ollama Actually Does](07-ollama-internals/01-what-ollama-actually-does.md)
- ⬜ [7.2 Model Files and Model Loading](07-ollama-internals/02-model-files-and-model-loading.md)
- ⬜ [7.3 Inference Server Architecture](07-ollama-internals/03-inference-server-architecture.md)
- ⬜ [7.4 HTTP API](07-ollama-internals/04-http-api.md)
- ⬜ [7.5 Model Execution Lifecycle](07-ollama-internals/05-model-execution-lifecycle.md)
- ⬜ [7.6 How Ollama Interacts with Local Hardware](07-ollama-internals/06-how-ollama-interacts-with-local-hardware.md)
- ⬜ [7.7 Ollama CLI Commands (ps, show and more)](07-ollama-internals/07-ollama-cli-commands-ps-show-and-more.md)
