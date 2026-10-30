"""Pin bounded non-LLM capture inputs. Native producer separately binds GGUF content."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from tools.igc.coherent_observation import _exclusive,_json_bytes
from tools.igc.coherent_collect import PROFILE_SHA,SERVER_SHA


def pin(prepared,profile,server,executable,bundle,pack,protocol,output):
    def sha(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    if sha(profile)!=PROFILE_SHA or sha(server)!=SERVER_SHA:raise ValueError('actual Hub profile/server binding')
    cfg=json.loads(profile.read_text(encoding='utf-8-sig'));cwd=Path(cfg.get('cwd') or '.')
    paths={profile,server,executable,bundle,prepared/'plan.bin',prepared/'manifest.json',protocol,Path(__file__)}
    paths.add(Path(sys.executable));paths.update(Path(sys.base_prefix).glob('python*.dll'))
    paths.update(Path(__file__).with_name(n) for n in ('coherent_collect.py','coherent_observation.py','coherent_prepare.py'))
    paths.update(f for f in pack.iterdir() if f.is_file())
    paths.update(f for f in (pack/'tokenizer').iterdir() if f.is_file())
    paths.update(bundle.parent/n for n in ('input-embedding.bin','input-attention.bin','readiness.bin','output.bin'))
    paths.update(executable.parent.glob('*.dll'))
    args=cfg['args']
    for flag in ('--mtp','--expert-profile'):
        if flag in args:
            p=Path(args[args.index(flag)+1]);p=p if p.is_absolute() else cwd/p
            if p.is_dir():paths.update(f for f in p.iterdir() if f.is_file() and f.suffix in ('.bin','.txt'))
            else:paths.add(p)
    for key in ('exe','model','mmproj'):
        if cfg.get('vision',{}).get(key):
            p=Path(cfg['vision'][key]);paths.add(p if p.is_absolute() else cwd/p)
    paths={p.resolve() for p in paths}
    if len(paths)>128 or sum(p.stat().st_size for p in paths)>12*1024**3:raise ValueError('bounded auxiliary source pinning exceeds12GiB/128files')
    files=[]
    for p in sorted(paths):
        before=p.stat();digest=sha(p);after=p.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('source changed while hashing')
        files.append(dict(path=str(p),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=digest))
    value=dict(verified=True,files=files,scope='current auxiliary inputs; actual native GGUF content bound by producer to plan before allocation',
               native_gguf_proof='not duplicated here; native VerifiedInputs binds actual loaded embedding/dense/head/PLE sources')
    _exclusive(output,_json_bytes(value))
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('prepared','profile','server','executable','bundle','pack','protocol','output'):p.add_argument('--'+key,type=Path,required=True)
    r=pin(**vars(p.parse_args()));print(json.dumps(dict(verified=True,files=len(r['files']),bytes=sum(x['bytes'] for x in r['files']),model_started=False)))


if __name__=='__main__':main()
