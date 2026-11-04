# Strata-xeno: MIP + Sandbox glossary

The words used for the work that makes a model hand its computation to software. A glossary only: what each word means, not how
it is built.

## The target

**Target model**: the model that MIP + Sandbox is built and measured on. Qwen3.6-35B-A3B in NVIDIA's NVFP4 release
(`C:\AI\models\nvidia-Qwen3.6-35B-A3B-NVFP4`), running on the strata-xeno engine. Its quality next to the bf16 model does not matter.
What matters is how much MIP + Sandbox raises its accuracy and lowers its tokens, measured against the same model with MIP off on the
same machine. (Developer, 2026-11-04.)

**Reference runner**: `tools/igc/tir4b.py` on llama-server or vLLM. A Python copy of the rules, used to try an idea quickly and to compare
against the engine. It is not the target: a result from it says what llama.cpp does, not what strata-xeno does.

## The mechanism

**MIP**: the part of the engine that notices the model computing and gives the work to software. It is one module for every model.
A new model connects to it; it is not rewritten for each model. It must not lower quality.

**Catch layer**: one of the five points where MIP can notice a computation, in the order they happen:
1. *before writing*: the hidden state says the model is about to compute (a probe);
2. *the expression begins*: the first operator of an expression;
3. *the expression is complete*: `=`, a code fence, or a tool call;
4. *after the answer*: the values written are checked;
5. *hand values*: a value the model computed itself is checked, and a wrong one is answered with the right one.

**Confirmed window**: the drafted tokens the model has verified (accepted) in a verify window but that have not been sent out
yet. They are certain, not guesses. MIP reads the whole window before it is sent, so a wrong hand value (`7 × 8 = 54`) can be cut at
its `=`, the state rolled back, and software's value put in. Neither the user nor the model ever sees the wrong value. It is a check
point for layers 3-5. It does not see before the model writes, only before the text goes out. (Developer, 2026-11-04.)

**Logit override**: how MIP hands over a value inside the engine. At the token where the model writes a value, the sampler's
choice is set to software's digits, so the model writes the right value as its own: no note, no correction after a wrong value. The
tokens are the same as an injected value; what differs is that the model never sees its own wrong value, and there is no extra request
round trip. It does not change what the model believes. (Developer's question, 2026-11-04.)

**Raw draft trend**: what the drafter proposes before the model verifies it, rejected proposals included. It is a guess, so it
only warns: a run of digits or operators in the proposals says the model is close to computing, and MIP gets ready (parser, software)
or counts it as evidence for layers 1-2. It never acts alone; the confirmed window or another layer decides. (Developer, 2026-11-04.)

**Whole computation**: what a catch layer asks for once it sees a computation coming: the model writes the whole calculation as one
expression (or one program, for a search over values) and software gives the value, before the model can cut it into small hand steps.
Layers 1 and 2 ask for it; layer 3 computes it. (Developer, 2026-11-04.)

**Hand computation**: a value the model wrote itself where software could have computed it. Every hand computation is one that the
catch layers before it missed.

**Hand-off**: the moment the model receives a value from software and goes on. A hand-off fails when the model does not use the value.
For example, it writes `.000...` after it, writes it again and again, or recomputes it by hand.

**Coverage**: what a catch layer caught out of what it should have caught, written as a fraction (8/10), never as a count of times it
fired. The denominator is always reported. Whatever no tool can count, such as arithmetic in the head with no expression written, is
named as not counted.
