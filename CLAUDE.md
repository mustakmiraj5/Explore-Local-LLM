# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal learning project: understanding how local LLMs work, using Ollama running `qwen3.5:4b` on macOS. Two parts:

- **Python experiments** at the repo root (`main.py`, `test-token.py`).
- **A self-study syllabus** in `docs/` (see below), which is the main focus of the work.

## Commands

```bash
source venv/bin/activate          # Python 3.14 venv
pip install -r requirements.txt

ollama serve                      # must be running; main.py talks to localhost:11434
ollama pull qwen3.5:4b

python main.py                    # interactive tool-calling chat; type 'exit' to quit
python test-token.py              # tiktoken encode/decode demo
```

There is no test suite, linter or build step.

## main.py architecture

A single-file tool-calling agent that uses the `openai` SDK against Ollama's OpenAI-compatible endpoint (`base_url=http://localhost:11434/v1`, dummy `api_key="ollama"`).

Adding a tool takes two edits, which must stay in sync:
1. Add the Python function and register it in `AVAILABLE_FUNCTIONS` (name → callable).
2. Add a matching JSON-schema entry to `TOOLS` (this is what the model sees).

`chat()` keeps the whole conversation in one `messages` list and allows up to 5 model↔tool rounds per user turn. Tool results are sent back as `role: "tool"` messages, JSON-encoded. `execute_tool` catches exceptions and returns them to the model as `{"error": ...}`. The date/time tool is hardcoded to `Asia/Dhaka`.

## docs/ syllabus

`docs/README.md` is the index and progress tracker for "LLM Fundamentals — Phase 1": 7 topics with numbered lessons (`docs/NN-topic/NN-lesson.md`). The status legend is ✅ done · 🟡 in progress · ⬜ not started.

- Lessons are written one at a time, only when the learner reaches them. Unstarted lessons are placeholder stubs (`> ⬜ Not started. This lesson is written when you reach it.`). Don't fill them in ahead of time.
- A finished lesson follows the structure of `01-llm-architecture/01-neural-network-fundamentals.md`: a header line (`Topic N: … · Lesson X of Y · Status: …`), then sections 1–8 (Introduction, Fundamentals, Under the Hood, Practical Implementation, Real-World Application, Common Misconceptions, Knowledge Assessment, Practical Assignment), and a closing `### Progress` checklist.
- Examples are grounded in this machine's real setup (Qwen 3.5 4B through Ollama, e.g. `ollama show --verbose` output) rather than generic models.
- When a lesson's status changes, update both its header line and its entry in `docs/README.md`.
