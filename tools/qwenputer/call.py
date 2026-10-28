"""Offline specification of native-call eligibility, without evaluating math."""
from dataclasses import dataclass
import re

from tools.qwenputer.tokenizer import NativeInputError


@dataclass(frozen=True)
class CallContext:
    live_assistant: bool = False
    enrolled_calc: bool = False
    code_fence: bool = False
    quoted: bool = False
    history: bool = False


def eligible_call(text, qids, context):
    if (not context.live_assistant or not context.enrolled_calc or context.code_fence
            or context.quoted or context.history):
        return None
    # The caller owns quote/fence/history state. This function never infers an
    # executable call from quoted text or guesses which external tool was meant.
    raw = text.decode_bytes(qids)
    if not raw.lstrip(b'\n').startswith(b'<function=calc>'):
        return None
    raw = text.capture_rows(qids)
    match = re.fullmatch(rb'\n?<function=calc><parameter=expression>([^<>]+)</parameter></function>\n?', raw)
    if match is None:
        raise NativeInputError('native_call_shape')
    expression = match[1].replace(b'\n', b'')
    if not 1 <= len(expression) <= 255:
        raise NativeInputError('expression_bytes')
    # No host evaluation. The firmware's parser and compiled circuit compute it.
    if not re.fullmatch(rb'[0-9+*/%^()., gcd\-]+', expression):
        raise NativeInputError('expression_syntax')
    return expression
