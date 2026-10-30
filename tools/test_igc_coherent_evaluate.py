"""Bound offline scorecards; native receipts below are literal synthetic fixtures."""
import hashlib
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from tools.test_igc_coherent_audit import fixture
from tools.igc.coherent_audit import audit_archive
from tools.igc.coherent_evaluate import evaluate, ledger_rows, evaluate_files
from tools.igc.coherent_observation import _json_bytes


def integer(digits,negative=0):
    return bytes([len(digits),negative,*digits])+bytes(62-len(digits))


def native_record(owner,choice,operation=0,left=None,right=None,result=None):
    active=choice==0
    raw=struct.pack('<QQ6I',owner,2,0,1,int(active),1 if active else 2 if choice==1 else 3,choice,0)
    raw+=struct.pack('<IIIIQQ',1,int(active),operation,0,owner,2)+(left+right if active else bytes(128))
    raw+=struct.pack('<IIQQ',1,0 if active else 4,owner,2)+(result if active else bytes(64))
    return (raw+struct.pack('<I',0 if active else 1)+bytes(2560*4)).hex()


class CoherentEvaluate(unittest.TestCase):
    def inputs(self,td):
        root=fixture(Path(td)/'archive');audit=audit_archive(root,require_native=False)
        rows=[dict(id='a',group=0,language='en',intent='imperative',lhs=17,rhs=23,operation=3,ready_op=3),
              dict(id='b',group=0,language='en',intent='quote-copy',lhs=17,rhs=23,operation=3,ready_op=0),
              dict(id='c',group=1,language='th',intent='question',lhs=-3,rhs=8,operation=2,ready_op=2),
              dict(id='d',group=1,language='th',intent='label',lhs=-3,rhs=8,operation=2,ready_op=0)]
        requests=[dict(ordinal=i,role=r['role'],base=r['base'],step=r['step'],sample=rows[r['base']]['id'],
                       target=7 if i in (0,4) else None) for i,r in enumerate(audit['row_receipts'])]
        manifest=dict(version=2,kind='coherent-development-manifest',plan_sha256=audit['plan_sha256'],
            rows=rows,requests=requests,anchors=4,positive=2,negative=2,clusters=[0,1],structural_clusters=2,
            acceptance64_access=False)
        primary=[[3,0,0,1,1,7,1,3,2],[0]*9,[2,1,0,0,0,3,0,8,0],[0]*9]
        native=[native_record(1,0,3,integer([7,1]),integer([3,2]),integer([1,9,3])),
                native_record(2,1),native_record(3,0,2,integer([3],1),integer([8]),integer([1,1],1)),
                native_record(4,1)]
        records=[dict(ordinal=i,owner=i+1,receipt_sha256=audit['row_receipts'][i]['receipt_sha256'],
            primary=fields,secondary=fields.copy(),readiness_logits=[0.,1.] if i in (0,2) else [1.,0.],
            record_hex=native[i]) for i,fields in enumerate(primary)]
        binding=dict(version=1,kind='coherent-native-replay',plan_sha256=audit['plan_sha256'],
            complete_sha256=audit['complete_sha256'],model_sha256=audit['model_sha256'],
            tokenizer_sha256=audit['tokenizer_sha256'],controller_sha256='55'*32,
            manifest_sha256=hashlib.sha256(_json_bytes(manifest)).hexdigest())
        return manifest,audit,binding,records

    def test_correct_tuples_calls_and_negative_bypasses_are_separate_from_output_use(self):
        with tempfile.TemporaryDirectory() as td:
            result=evaluate(*self.inputs(td))
            self.assertEqual(result['counts']['selected_correct'],2)
            self.assertEqual(result['counts']['negative_calls'],0)
            self.assertEqual(result['readouts']['primary']['positive_exact'],2)
            self.assertEqual(result['counts']['readiness_correct'],4)
            self.assertTrue(result['input_gates_pass'])
            self.assertFalse(result['live_or_acceptance_qualified'])
            self.assertEqual(result['hard_off_output']['positions'],2)
            self.assertEqual(result['hard_off_output']['target_top1'],0)
            self.assertEqual(result['cluster_bootstrap']['complete_group_rate_interval95'],[1.,1.])
            self.assertEqual(result['output_controls'],{'learned':'not supplied','oracle_owned':'not supplied'})

    def test_readiness_miss_and_padding_disagreement_remain_distinct_raw_failures(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            records[0]['readiness_logits']=[1.,0.];records[0]['record_hex']=native_record(1,1)
            records[2]['primary'][6]=3;records[2]['record_hex']=native_record(3,4)
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['counts']['selected_correct'],0)
            self.assertEqual(result['positive_partition']['readiness_only'],1)
            self.assertEqual(result['positive_partition']['primary_only'],1)
            self.assertEqual(result['readouts']['primary']['padding_only_error'],1)
            self.assertEqual(result['padding_composition_counterfactual']['selected_correct'],1)
            self.assertFalse(result['input_gates_pass'])

    def test_quoted_false_call_cannot_be_hidden_by_correct_arithmetic(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            records[1].update(primary=records[0]['primary'].copy(),secondary=records[0]['secondary'].copy(),
                readiness_logits=[0.,1.],record_hex=native_record(2,0,3,integer([7,1]),integer([3,2]),integer([1,9,3])))
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['counts']['negative_calls'],1)
            self.assertEqual(result['counts']['negative_readiness_false_positive'],1)
            self.assertFalse(result['input_gates_pass'])

    def test_active_digit_errors_survive_the_separate_padding_counterfactual(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            records[2]['primary'][5]=4;records[2]['primary'][6]=3
            records[2]['secondary']=records[2]['primary'].copy();records[2]['record_hex']=native_record(3,5)
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['readouts']['primary']['active_and_padding_error'],1)
            self.assertEqual(result['padding_composition_counterfactual']['selected_correct'],1)
            confusion=[r for r in result['readouts']['primary']['digit_confusions']
                       if r['slot']=='left_ones' and r['expected']==3 and r['predicted']==4]
            self.assertEqual(confusion,[dict(slot='left_ones',expected=3,predicted=4,width=1,padding=False,count=1)])
            self.assertFalse(result['input_gates_pass'])

    def test_frozen_labels_row_ownership_and_typed_results_cannot_be_reassigned(self):
        with tempfile.TemporaryDirectory() as td:
            original=self.inputs(td)
            for kind in ('manifest','missing','extra','order','owner','teacher-role','bad-result','nonfinite'):
                manifest,audit,binding,records=copy.deepcopy(original)
                if kind=='manifest':manifest['rows'][0]['lhs']=18
                elif kind=='missing':records.pop()
                elif kind=='extra':records.append(records[-1].copy())
                elif kind=='order':records[0],records[1]=records[1],records[0]
                elif kind=='owner':records[0]['owner']=777
                elif kind=='teacher-role':records[0]['ordinal']=4
                elif kind=='nonfinite':records[0]['readiness_logits']=[0.,float('nan')]
                else:
                    raw=bytearray.fromhex(records[0]['record_hex']);raw[226]=2
                    records[0]['record_hex']=raw.hex()
                with self.subTest(kind=kind),self.assertRaises(ValueError):
                    evaluate(manifest,audit,binding,records)

    def test_streamed_ledger_requires_newline_bounded_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'ledger.jsonl'
            path.write_bytes(b'{"version":1}\n{"ordinal":0}\n')
            self.assertEqual(list(ledger_rows(path)),[{'version':1},{'ordinal':0}])
            for raw in (b'{"version":1}',b'{}\n{}',b'{}\n'+b' '*32768+b'\n'):
                path.write_bytes(raw)
                with self.subTest(size=len(raw)),self.assertRaises(ValueError):list(ledger_rows(path))

    def test_valid_execution_of_wrong_operands_is_still_a_task_error(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            records[0]['primary'][5]=8;records[0]['secondary']=records[0]['primary'].copy()
            records[0]['record_hex']=native_record(1,0,3,integer([8,1]),integer([3,2]),integer([4,1,4]))
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['counts']['selected_wrong'],1)
            self.assertEqual(result['positive_partition']['both_readouts'],1)
            self.assertEqual(result['rows'][0]['native_result'],414)
            self.assertFalse(result['input_gates_pass'])

    def test_file_interface_is_read_only_bound_and_refuses_report_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);manifest,audit,binding,records=self.inputs(td)
            manifest_path=root/'manifest.json';manifest_path.write_bytes(_json_bytes(manifest))
            replay=root/'replay.jsonl';replay.write_bytes(_json_bytes(binding)+b''.join(_json_bytes(r) for r in records))
            before=(manifest_path.read_bytes(),replay.read_bytes());output=root/'report.json'
            result=evaluate_files(root/'archive',manifest_path,replay,output,allow_synthetic=True)
            self.assertEqual(result['counts']['selected_correct'],2)
            self.assertEqual(json.loads(output.read_text())['inputs']['replay_sha256'],hashlib.sha256(before[1]).hexdigest())
            self.assertEqual(before,(manifest_path.read_bytes(),replay.read_bytes()))
            with self.assertRaises(FileExistsError):evaluate_files(root/'archive',manifest_path,replay,output,allow_synthetic=True)
            with self.assertRaisesRegex(ValueError,'native'):evaluate_files(root/'archive',manifest_path,replay,root/'native.json')
            with self.assertRaisesRegex(ValueError,'outside'):evaluate_files(root/'archive',manifest_path,replay,root/'archive'/'report.json',allow_synthetic=True)

    def test_relabelled_teacher_forced_target_is_refused_after_hash_binding(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            manifest['requests'][4]['target']=8
            binding['manifest_sha256']=hashlib.sha256(_json_bytes(manifest)).hexdigest()
            with self.assertRaisesRegex(ValueError,'target alignment'):
                evaluate(manifest,audit,binding,iter(records))

    def test_controller_identity_is_bound_to_archived_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td);binding['controller_sha256']='66'*32
            with self.assertRaisesRegex(ValueError,'controller'):
                evaluate(manifest,audit,binding,records)

    def test_readiness_bypass_does_not_hide_raw_disagreement_or_invalid_padding(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            records[2]['primary'][6]=3;records[2]['readiness_logits']=[1.,0.]
            records[2]['record_hex']=native_record(3,1)
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['counts']['raw_disagreement'],1)
            self.assertEqual(result['counts']['raw_primary_invalid_encoding'],1)
            self.assertEqual(result['counts']['raw_secondary_invalid_encoding'],0)
            self.assertEqual(result['counts']['routed_disagreement'],0)
            self.assertEqual(result['counts']['routed_invalid_encoding'],0)

    def test_matched_negative_cells_keep_their_expression_axes(self):
        with tempfile.TemporaryDirectory() as td:
            result=evaluate(*self.inputs(td))
            self.assertEqual(result['by_condition']['operation:3']['negative'],1)
            self.assertEqual(result['by_condition']['width:2/2']['negative'],1)
            self.assertEqual(result['by_condition']['sign:1/0']['negative'],1)

    def test_ordinary_negative_does_not_invent_an_operand_cell(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            manifest['rows'][3]=dict(id='d',group=None,language='th',intent='ordinary',ready_op=0)
            binding['manifest_sha256']=hashlib.sha256(_json_bytes(manifest)).hexdigest()
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['by_condition']['expression:unclassified ordinary/unsupported']['negative'],1)
            self.assertEqual(result['by_condition']['operation:2']['negative'],0)

    def test_resampling_uses_structural_clusters_instead_of_prompt_or_digit_count(self):
        with tempfile.TemporaryDirectory() as td:
            manifest,audit,binding,records=self.inputs(td)
            manifest.update(clusters=[0,0],structural_clusters=1)
            binding['manifest_sha256']=hashlib.sha256(_json_bytes(manifest)).hexdigest()
            result=evaluate(manifest,audit,binding,records)
            self.assertEqual(result['cluster_bootstrap']['units'],1)
            self.assertEqual(result['cluster_bootstrap']['groups'],2)


if __name__=='__main__':unittest.main()
