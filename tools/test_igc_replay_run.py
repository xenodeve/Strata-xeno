"""Pricing may admit one fixed fit only after complete measured replay coverage."""
import unittest
import tempfile
from pathlib import Path
from tools.igc.replay_run import estimate_budget,verify_price,current_admission,read_receipt


class ReplayRun(unittest.TestCase):
    def test_budget_is_conservative_and_refuses_incomplete_or_expensive_price(self):
        r=estimate_budget(50,[2.0]+[.2]*7,30,840)
        self.assertGreaterEqual(r['fit_estimate_seconds'],2*50+4*132.99816550000105+60)
        self.assertTrue(r['admitted'])
        with self.assertRaises(ValueError):estimate_budget(50,[2.0]+[.2]*7,30,839)
        self.assertFalse(estimate_budget(100,[3]+[2]*7,50,840)['admitted'])

    def test_price_binding_includes_complete_schedule_sources_reset_and_no_steps(self):
        binding=dict(initial_state_sha256='init',first_batch_ids=['train'],source_sha256={'x':'a'},
            cache_identity={'old':'a','matched':'b'},protocol_sha256='protocol',recipe={'updates':840},replay_sha256='replay')
        receipt={**binding,'success':True,'optimizer_steps':0,'backward_batches':8,'gather_batches':840,
            'optimizer_constructed':False,
            'io_total_seconds':50,'backward_seconds':[2.0]+[.2]*7,'initial_validation_seconds':30,
            'estimate':estimate_budget(50,[2.0]+[.2]*7,30,840)}
        verify_price(receipt,binding)
        for field,value in [('optimizer_steps',1),('backward_batches',7),('gather_batches',839),('replay_sha256','changed'),
                            ('cache_identity',{'old':'different'}),('recipe',{'updates':841}),('estimate',{'admitted':False}),
                            ('io_total_seconds',51)]:
            with self.subTest(field=field),self.assertRaises(ValueError):verify_price({**receipt,field:value},binding)

    def test_fit_rechecks_current_validation_and_actual_outer_prefix_before_updates(self):
        receipt={'io_total_seconds':20,'backward_seconds':[2]+[.83]*7,'initial_validation_seconds':90,'gather_batches':840}
        self.assertTrue(current_admission(receipt,90,105)['admitted'])
        self.assertFalse(current_admission(receipt,250,260)['admitted'])
        self.assertFalse(current_admission(receipt,90,300)['admitted'])

    def test_price_reader_refuses_oversized_file_before_parsing(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'price.json';p.write_bytes(b'0'*(256*1024+1))
            with self.assertRaises(ValueError):read_receipt(p)


if __name__=='__main__':unittest.main()
