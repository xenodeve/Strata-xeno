"""SEAL (Steerable Reasoning Calibration, arXiv 2504.07986) for the reference runner: a steering vector that holds the reasoning back from reflecting and switching
and keeps it executing, added to the state at every thought separator while the model generates. Training-free; nothing is cut and no text is put in.

What the paper states (read in full, v3): a thought is a piece between two `\\n\\n`; it is labelled by keywords only (table 5); the representation is the output of block i
at the thought's `\\n\\n` token; S = mean over execution thoughts - mean over reflection + transition thoughts; at inference H~ = H + alpha * S at every `\\n\\n`
token the model writes; alpha = 1, layer 20 for 1.5B and 7B models; the vector comes from 1,000 training samples the target model itself wrote. Not stated: whether the
prompt's `\\n\\n` tokens are steered, and the keyword priority when a thought matches both lists (here: transition first).

Here: the state is the stream after block L (1-based, L blocks done): in vLLM that is hidden_states + residual of layers[L-1]'s output, and the delta is added to
hidden_states, which the next layer adds to the residual. The prompt's `\\n\\n` tokens are steered too (one flat batch cannot tell the prompt from the reply
without the scheduler's metadata); the effect is small and is this file's one departure from the paper.
"""
import re

TRANSITION_PREFIX = ('alternatively',)
TRANSITION_PHRASES = ('think differently', 'think differenly', 'another way', 'another approach', 'another method', 'another solution', 'another strategy',
                      'another technique')
REFLECTION_PREFIX = ('wait',)
REFLECTION_PHRASES = ('verify', 'make sure', 'hold on', 'think again', "'s correct", "'s incorrect", 'let me check', 'seems right')


def thought_type(text):
    """'E' execution, 'R' reflection, 'T' transition, by the paper's keywords (case-insensitive); transition wins a tie."""
    t = text.strip().lower()
    if t.startswith(TRANSITION_PREFIX) or any(p in t for p in TRANSITION_PHRASES):
        return 'T'
    if t.startswith(REFLECTION_PREFIX) or any(p in t for p in REFLECTION_PHRASES):
        return 'R'
    return 'E'


def thoughts(token_ids, tok, delim_id):
    """[(first token index, index of the delimiter that ends it, kind)] for every thought of a reply that is followed by a delimiter; the last, unfinished piece is dropped."""
    out, start = [], 0
    for i, t in enumerate(token_ids):
        if t == delim_id:
            out.append((start, i, thought_type(tok.decode(token_ids[start:i]))))
            start = i + 1
    return out


def find_modules(model, n_layers):
    """(embed_tokens module, the ModuleList of decoder layers) inside a vLLM or HF model: searched by name and length, so the nesting of a text-only or a vision-language wrapper does not matter."""
    import torch
    embed = layers = None
    for name, m in model.named_modules():
        if name.endswith('embed_tokens') and embed is None:
            embed = m
        if name.endswith('layers') and isinstance(m, torch.nn.ModuleList) and len(m) == n_layers and layers is None:
            layers = m
    if embed is None or layers is None:
        raise RuntimeError('SEAL: no embed_tokens or no layer list of the expected length in the model')
    return embed, layers


def install_steering(model, vector, layer, alpha, delim_id, n_layers):
    """Hooks on a model (vLLM worker side or HF): the input ids are read at embed_tokens; the output of block `layer` (1-based) gets alpha * vector added at every position whose input
    token is `delim_id`. Returns the hook handles (remove() to undo) and a counter dict {'steered': n} of how many positions were steered, for the run's log."""
    import torch
    embed, layers = find_modules(model, n_layers)
    held, count = {}, {'steered': 0, 'calls': 0}

    def grab(_m, args, kwargs=None):
        held['ids'] = args[0] if args else (kwargs or {}).get('input_ids')

    def steer(_m, _args, output):
        ids, held['ids'] = held.get('ids'), None   # read once: a forward that did not pass through embed_tokens (input embeddings given) must not reuse the last call's ids
        h = output[0] if isinstance(output, tuple) else output
        if ids is None or ids.shape[0] != h.shape[0]:
            return output
        mask = ids == delim_id
        n = int(mask.sum())
        count['calls'] += 1
        if n:
            h[mask] += (alpha * vector).to(h.dtype).to(h.device)
            count['steered'] += n
        return output

    return [embed.register_forward_pre_hook(grab), layers[layer - 1].register_forward_hook(steer)], count
