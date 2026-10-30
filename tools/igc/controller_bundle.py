"""Build a version-2 controller bundle: attention alone selects, with a refitted readiness head."""
import argparse
import hashlib
from pathlib import Path
import shutil
import struct
import torch
from safetensors.torch import load_file
from tools.igc.adapter_artifact import pack_state


def build(source,readiness_state,protocol,output):
    raw=(source/'controller.bin').read_bytes()
    if len(raw)!=204 or raw[:8]!=b'IGCLAT01':raise ValueError('controller bundle source')
    model,tokenizer=raw[12:44],raw[44:76]
    state=hashlib.sha256(readiness_state.read_bytes()).hexdigest()
    packed=pack_state(3,dict(load_file(str(readiness_state))),model.hex(),tokenizer.hex(),state,hashlib.sha256(protocol.read_bytes()).hexdigest())
    output.mkdir(exist_ok=False)
    for name in ('input-embedding.bin','input-attention.bin','output.bin'):shutil.copyfile(source/name,output/name)
    (output/'readiness.bin').write_bytes(packed)
    digests=[hashlib.sha256((output/n).read_bytes()).digest() for n in ('input-embedding.bin','input-attention.bin','readiness.bin','output.bin')]
    (output/'controller.bin').write_bytes(b'IGCLAT01'+struct.pack('<I',2)+model+tokenizer+b''.join(digests))
    return hashlib.sha256((output/'controller.bin').read_bytes()).hexdigest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('source','readiness-state','protocol','output'):parser.add_argument('--'+key,type=Path,required=True)
    print(build(**vars(parser.parse_args())))
