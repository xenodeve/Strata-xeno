"""Replay preparation must exclude semantic evaluation overlap before scheduling."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter
from unittest.mock import patch
from tools.igc.replay_protocol import exclusion_keys,filter_training,schedule,validate_schedule,prepared_payload,verify_prepared,prepare,presentation_histogram


def row(identifier,a,b,symbol='+',split='train',ready=None,prompt=None):
    return dict(id=identifier,lhs=a,rhs=b,symbol=symbol,split=split,
        ready_op={'+':1,'-':2,'*':3}[symbol] if ready is None else ready,
        expression=f'({a}) {symbol} ({b})',prompt=prompt or f'Compute ({a}) {symbol} ({b}).')


class ReplayProtocol(unittest.TestCase):
    def test_expression_is_parsed_and_must_match_metadata_even_for_negative(self):
        a=row('a',-1,97,ready=0);self.assertEqual(exclusion_keys(a)['expression'],(-1,'+',97))
        for bad in ['(-1) + (7)','__import__("os")','(-1 + 97)']:
            with self.assertRaises(ValueError):exclusion_keys({**a,'expression':bad})

    def test_fact_normalization_covers_reordering_subtraction_and_multiply_signs(self):
        key=lambda a,b,s:exclusion_keys(row('x',a,b,s))['fact']
        self.assertEqual(key(17,23,'+'),key(23,17,'+'))
        self.assertEqual(key(17,23,'-'),key(17,-23,'+'))
        self.assertEqual(key(-3,-8,'*'),key(8,3,'*'))
        self.assertNotEqual(key(1,2,'+'),key(1,3,'*'))  # Equal answers alone are not duplicates.

    def test_filter_never_selects_eval_rows_or_swapped_pair_or_fact_or_prompt(self):
        held=[row('held',17,23,split='development')]
        candidates=[row('samepair',17,23,'-'),row('swapped',23,17,'-'),
            row('fact',17,-23,'-'),row('prompt',1,2,prompt='  COMPUTE (17) + (23).  '),
            row('keep',4,9),row('nott',4,8,split='development')]
        chosen,excluded=filter_training(candidates,held)
        self.assertEqual([r['id'] for r in chosen],['keep'])
        self.assertIn('symmetric_pair',excluded['swapped'])
        self.assertIn('fact',excluded['fact']);self.assertIn('prompt',excluded['prompt'])
        self.assertIn('not_train',excluded['nott'])

    def test_schedule_is_deterministic_32_by_32_and_old_classes_are_fixed(self):
        members=[{'source':'matched','index':i,'id':f'm{i}'} for i in range(32)]
        members += [{'source':'old','index':i,'id':f'o{i}'} for i in range(4)]
        old={op:[32+op] for op in range(4)}
        a=schedule(list(range(32)),old,updates=2);b=schedule(list(range(32)),old,updates=2)
        self.assertEqual(a,b);validate_schedule(a,members,list(range(32)),old,updates=2)
        for batch in a:
            self.assertEqual(set(batch[:32]),set(range(32)))
            self.assertEqual(Counter(batch[32:]),Counter({32:8,33:8,34:8,35:8}))
        bad=copy.deepcopy(a);bad[0][0]=32
        with self.assertRaises(ValueError):validate_schedule(bad,members,list(range(32)),old,updates=2)
        bad=copy.deepcopy(a);bad[0][-1]=999
        with self.assertRaises(ValueError):validate_schedule(bad,members,list(range(32)),old,updates=2)

    def test_input_and_reviewed_schedule_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            paths=[Path(root)/str(i) for i in range(3)]
            for p in paths:p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'frozen replay inputs'):prepared_payload(*paths)
            p=Path(root)/'replay';payload={'members':[],'batches':[]};p.write_text(json.dumps(payload))
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            with patch('tools.igc.replay_protocol.prepared_payload',return_value=payload):
                self.assertEqual(verify_prepared(p,*paths,digest),payload)
                with self.assertRaises(ValueError):verify_prepared(p,*paths,'0'*64)
            with patch('tools.igc.replay_protocol.prepared_payload',return_value={**payload,'batches':[[1]]}):
                with self.assertRaises(ValueError):verify_prepared(p,*paths,digest)

    def test_prepare_json_verify_preserves_real_presentation_histogram(self):
        payload={'presentation_counts':{'old_each_histogram':presentation_histogram([2,3,2])}}
        self.assertEqual(payload['presentation_counts']['old_each_histogram'],{'2':2,'3':1})
        with tempfile.TemporaryDirectory() as root,patch('tools.igc.replay_protocol.prepared_payload',return_value=payload):
            out=Path(root)/'new';prepare('old','matched','selection',out)
            p=out/'replay.json';digest=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(verify_prepared(p,'old','matched','selection',digest),payload)


if __name__=='__main__':unittest.main()
