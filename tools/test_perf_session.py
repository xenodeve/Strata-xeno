"""Measurement completeness gates from small files, no engine or GPU."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'tests/xeno/perf/ab.py'
spec=importlib.util.spec_from_file_location('strata_perf_session_test',path)
ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)


class PerfSession(unittest.TestCase):
    def complete(self,root,order=None,prompts=None):
        order=order or ['A','B','B','A'];arms={}
        prompts=prompts or ['code']
        for arm in sorted(set(order)):
            raw=('fixture '+arm).encode();(root/(arm+'.exe')).write_bytes(raw)
            arms[arm]=hashlib.sha256(raw).hexdigest()
        (root/'session.json').write_text(json.dumps(dict(version=1,arms=arms,prompts=prompts,order=order)))
        for prompt in prompts:
            for i,arm in enumerate(order,1):
                tag=f'{prompt}-{i}-{arm}'
                (root/(tag+'.stdout')).write_text('output : 1 2 3\n-> 10.00 tok/s\n')
                (root/(tag+'.stderr')).write_text('fixture stderr')
                (root/(tag+'.run.json')).write_text(json.dumps(dict(version=1,prompt=prompt,ordinal=i,arm=arm,
                    exe_sha256=arms[arm],returncode=0,memory_complete=True,
                    stdout_sha256=hashlib.sha256((root/(tag+'.stdout')).read_bytes()).hexdigest(),
                    stderr_sha256=hashlib.sha256((root/(tag+'.stderr')).read_bytes()).hexdigest())))

    def test_one_legacy_arm_cannot_qualify_cross_arm_parity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'code-1-A.stdout').write_text('output : 1 2 3\n')
            summary=ab.summarize(str(root),['code'])
            self.assertNotIn('outputs identical across arms: True',summary)
            self.assertIn('unverified',summary)

    def test_complete_session_qualifies_parity_and_preserves_stage_tables(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.complete(root)
            evidence=ab.session_evidence(str(root),'code')
            self.assertTrue(evidence['complete']);self.assertTrue(evidence['parity_qualified'])
            self.assertTrue(evidence['memory_complete'])
            self.assertIn('outputs identical across arms: True',ab.summarize(str(root),['code']))

    def test_missing_failed_mismatched_or_extra_evidence_cannot_qualify(self):
        for case in ('missing','failed','snapshot','receipt','run-version','stderr','extra-receipt','empty'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as td:
                root=Path(td);self.complete(root)
                if case=='missing':(root/'code-4-A.stdout').unlink()
                elif case=='snapshot':(root/'A.exe').write_bytes(b'changed')
                elif case=='stderr':(root/'code-4-A.stderr').unlink()
                elif case=='extra-receipt':(root/'code-5-A.run.json').write_text('{}')
                elif case=='empty':(root/'code-4-A.stdout').write_text('output :\n')
                else:
                    path=root/'code-4-A.run.json';row=json.loads(path.read_text())
                    row['returncode']=1 if case=='failed' else 0
                    if case=='receipt':row['exe_sha256']='00'*32
                    if case=='run-version':row['version']=True
                    path.write_text(json.dumps(row))
                self.assertFalse(ab.session_evidence(str(root),'code')['parity_qualified'])

    def test_token_difference_and_memory_failure_are_separate_from_completeness(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.complete(root)
            path=root/'code-1-A.run.json';row=json.loads(path.read_text());row['memory_complete']=False
            path.write_text(json.dumps(row));evidence=ab.session_evidence(str(root),'code')
            self.assertTrue(evidence['complete']);self.assertTrue(evidence['parity_qualified']);self.assertFalse(evidence['memory_complete'])
            (root/'code-4-A.stdout').write_text('output : 1 2 4\n')
            path=root/'code-4-A.run.json';row=json.loads(path.read_text())
            row['stdout_sha256']=hashlib.sha256((root/'code-4-A.stdout').read_bytes()).hexdigest();path.write_text(json.dumps(row))
            evidence=ab.session_evidence(str(root),'code')
            self.assertTrue(evidence['complete']);self.assertFalse(evidence['parity_qualified'])

    def test_one_planned_arm_and_reused_output_directory_are_not_cross_arm_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.complete(root,['A','A'])
            evidence=ab.session_evidence(str(root),'code')
            self.assertTrue(evidence['complete']);self.assertFalse(evidence['parity_qualified'])
            before=(root/'session.json').read_bytes()
            with patch.object(ab,'T',str(root.parent)),patch.object(ab.subprocess,'Popen',side_effect=AssertionError('no launch')):
                with self.assertRaisesRegex(ValueError,'fresh'):
                    ab.main([root.name,'A=not-used.exe','B=not-used.exe'])
            self.assertEqual((root/'session.json').read_bytes(),before)

    def test_invalid_names_are_rejected_before_snapshot_copy(self):
        with tempfile.TemporaryDirectory() as td:
            with patch.object(ab,'T',td),patch.object(ab.shutil,'copy2',side_effect=AssertionError('no copy before validation')):
                with self.assertRaises(ValueError):ab.main(['run','../outside=not-used.exe','B=not-used.exe'])

    def test_replaced_matching_log_cannot_turn_divergent_evidence_into_parity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.complete(root)
            stdout=root/'code-4-A.stdout';stdout.write_text('output : 9 9\n')
            receipt=root/'code-4-A.run.json';row=json.loads(receipt.read_text())
            row['stdout_sha256']=hashlib.sha256(stdout.read_bytes()).hexdigest();receipt.write_text(json.dumps(row))
            stdout.write_bytes((root/'code-1-A.stdout').read_bytes())
            self.assertFalse(ab.session_evidence(str(root),'code')['parity_qualified'])

    def test_missing_or_failed_other_prompt_prevents_whole_session_qualification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.complete(root,prompts=['code','thai'])
            self.assertTrue(ab.session_evidence(str(root),'code')['parity_qualified'])
            stdout=root/'thai-1-A.stdout';original=stdout.read_bytes();stdout.unlink()
            self.assertFalse(ab.session_evidence(str(root),'code')['complete'])
            stdout.write_bytes(original)
            receipt=root/'thai-1-A.run.json';row=json.loads(receipt.read_text());row['returncode']=1
            receipt.write_text(json.dumps(row))
            self.assertFalse(ab.session_evidence(str(root),'code')['parity_qualified'])


if __name__=='__main__':unittest.main()
