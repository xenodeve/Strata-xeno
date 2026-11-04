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
1b. *draft lookahead*: the drafter (MTP, or any draft model) proposes the next tokens, and they are numbers, operators or a code
   fence. It needs no calibration for each model, and it sees one or two tokens before layer 2. Measured beside 1 and 2 before it is
   used (developer's question, 2026-11-04);
2. *the expression begins*: the first operator of an expression;
3. *the expression is complete*: `=`, a code fence, or a tool call;
4. *after the answer*: the values written are checked;
5. *hand values*: a value the model computed itself is checked, and a wrong one is answered with the right one.

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
