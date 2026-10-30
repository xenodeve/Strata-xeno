"""Small synthetic archives; never open model inputs or acceptance corpora."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from tools.igc.coherent_observation import CompactArchive, encode_plan
from tools.igc.coherent_audit import audit_archive


def fixture(root):
    records = [dict(role='anchor', base=i, step=0, tokens=[1, 2, 3+i]) for i in range(4)]
    records.append(dict(role='output', base=0, step=1, tokens=[1, 2, 3, 7]))
    plan = encode_plan(dict(model='11'*32, tokenizer='22'*32, records=records))
    writer = CompactArchive(root, plan, dict(engine='33'*32, profile='44'*32,bundle='55'*32))
    states = np.zeros((80,4,2560),dtype='<f4')
    states[-2] = 2; states[-1] = 4
    for i, record in enumerate(records):
        output = record['role'] == 'output'
        tokens = record['tokens']
        raw = struct.pack('<8s7IQ',b'IGCCOH03',3,int(output),i,record['base'],record['step'],len(tokens),0 if output else 2,i+1)
        raw += bytes.fromhex('11'*32+'22'*32)+plan[-32:]
        raw += struct.pack('<4Q6I',0,len(tokens),i,i+1,0,0,1,31,0,0)
        raw += struct.pack('<'+'I'*len(tokens),*tokens)
        raw += bytes(80) if output else bytes(78)+b'\1\1'
        if not output: raw += states.tobytes()*2
        raw += np.ones(2560,dtype='<f4').tobytes()+np.zeros(248320,dtype='<f4').tobytes()
        raw += hashlib.sha256(raw).digest()
        writer.ingest(raw,i,target=7 if i in (0,4) else None)
    writer.finalize()
    return root


class CoherentAudit(unittest.TestCase):
    def test_whole_archive_checks_all_rows_without_qualifying_synthetic_data(self):
        with tempfile.TemporaryDirectory() as td:
            root=fixture(Path(td)/'archive')
            result=audit_archive(root,require_native=False)
            self.assertEqual((result['rows'],result['anchors'],result['outputs']),(5,4,1))
            self.assertFalse(result['native_qualified'])
            self.assertEqual(result['source_counts'],{'synthetic':5,'native_verifier':0})
            self.assertEqual([r['owner'] for r in result['row_receipts']],[1,2,3,4,5])
            self.assertNotIn('attention',result['row_receipts'][0])
            with self.assertRaisesRegex(ValueError,'native'):
                audit_archive(root)

    def test_unreferenced_dictionary_rows_are_verified_too(self):
        with tempfile.TemporaryDirectory() as td:
            root=fixture(Path(td)/'archive')
            dictionary=json.loads((root/'dictionary.json').read_text())
            dictionary.append(dict(index=2,token=999,sha256='aa'*32))
            raw=(json.dumps(dictionary)+'\n').encode()
            (root/'dictionary.json').write_bytes(raw)
            with (root/'embeddings.bin').open('ab') as table:table.write(bytes(4*2560*4))
            complete=json.loads((root/'COMPLETE.json').read_text())
            complete.update(embedding_rows=3,dictionary_sha256=hashlib.sha256(raw).hexdigest(),
                embedding_sha256=hashlib.sha256((root/'embeddings.bin').read_bytes()).hexdigest())
            (root/'COMPLETE.json').write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError,'dictionary row'):
                audit_archive(root,require_native=False)

    def test_partial_markers_and_resealed_semantic_faults_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=fixture(Path(td)/'archive')
            pending=root/'unfinished.pending';pending.write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'incomplete'):
                audit_archive(root,require_native=False)
            pending.unlink()
            row_raw=(root/'0.json').read_bytes();complete_raw=(root/'COMPLETE.json').read_bytes()
            for change in ('argmax','logsumexp','target_logit','base_owner'):
                row=json.loads(row_raw);complete=json.loads(complete_raw)
                if change=='base_owner':row[change]=999
                elif change=='argmax':row['head'][change]=True
                elif change=='logsumexp':row['head'][change]+=1
                else:row['head'][change]=float('nan')
                raw=json.dumps(row).encode();(root/'0.json').write_bytes(raw)
                complete['rows'][0]=hashlib.sha256(raw).hexdigest()
                (root/'COMPLETE.json').write_text(json.dumps(complete))
                with self.subTest(change=change),self.assertRaises(ValueError):
                    audit_archive(root,require_native=False)
            (root/'0.json').write_bytes(row_raw);(root/'COMPLETE.json').write_bytes(complete_raw)
            self.assertEqual(audit_archive(root,require_native=False)['rows'],5)

    def test_complete_table_hash_is_required_even_when_referenced_rows_are_valid(self):
        with tempfile.TemporaryDirectory() as td:
            root=fixture(Path(td)/'archive')
            complete=json.loads((root/'COMPLETE.json').read_text())
            complete['embedding_sha256']='ff'*32
            (root/'COMPLETE.json').write_text(json.dumps(complete))
            with self.assertRaisesRegex(ValueError,'complete embedding digest'):
                audit_archive(root,require_native=False)

    def test_resealed_row_requires_v3_and_source_capture_digest(self):
        with tempfile.TemporaryDirectory() as td:
            root=fixture(Path(td)/'archive')
            original=(root/'0.json').read_bytes();complete_raw=(root/'COMPLETE.json').read_bytes()
            for key,value in [('version',99),('source_capture_sha256','not-a-digest')]:
                row=json.loads(original);row[key]=value;raw=json.dumps(row).encode()
                (root/'0.json').write_bytes(raw)
                complete=json.loads(complete_raw);complete['rows'][0]=hashlib.sha256(raw).hexdigest()
                (root/'COMPLETE.json').write_text(json.dumps(complete))
                with self.subTest(key=key),self.assertRaises(ValueError):
                    audit_archive(root,require_native=False)


if __name__=='__main__':unittest.main()
