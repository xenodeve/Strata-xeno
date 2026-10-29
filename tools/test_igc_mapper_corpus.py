"""Frozen splits for the first activation-readout feasibility experiment."""
import collections
import unittest
from tools.igc.mapper_corpus import corpus, evaluation_manifest


class MapperCorpus(unittest.TestCase):
    def test_new_evaluation_is_frozen_separate_and_never_marked_training(self):
        evaluation=evaluation_manifest()
        self.assertEqual(evaluation,evaluation_manifest())
        rows=evaluation['rows']
        self.assertEqual(len(rows),64)
        self.assertEqual({r['split'] for r in rows},{'evaluation'})
        self.assertEqual(collections.Counter(r['ready_op'] for r in rows),{op:16 for op in range(4)})
        self.assertFalse({r['prompt'] for r in rows}&{r['prompt'] for r in corpus()})
        self.assertFalse({(r['lhs'],r['rhs']) for r in rows}&{(r['lhs'],r['rhs']) for r in corpus()})
        self.assertEqual(len({r['prompt'] for r in rows}),64)

    def test_frozen_split_balance_and_no_prompt_leakage(self):
        rows=corpus()
        self.assertEqual(rows,corpus())
        self.assertEqual(len(rows),64)
        self.assertEqual(len({r['id'] for r in rows}),64)
        self.assertEqual(len({r['prompt'] for r in rows}),64)
        counts=collections.Counter((r['split'],r['ready_op']) for r in rows)
        self.assertEqual(counts,{(split,op):n for split,n in [('train',12),('heldout',4)] for op in range(4)})
        train={r['template'] for r in rows if r['split']=='train'}
        heldout={r['template'] for r in rows if r['split']=='heldout'}
        self.assertFalse(train & heldout)

    def test_every_label_is_bound_to_the_literal_expression(self):
        for r in corpus():
            self.assertIn(r['expression'],r['prompt'])
            self.assertEqual(r['expression'],f"{r['lhs']}{r['symbol']}{r['rhs']}")
            self.assertEqual(r['operand_scope'],'signed integer magnitude <= 99')
            self.assertTrue(-99 <= r['lhs'] <= 99 and -99 <= r['rhs'] <= 99)
            if r['ready_op']:
                self.assertEqual(r['symbol'],{1:'+',2:'-',3:'*'}[r['ready_op']])


if __name__=='__main__':
    unittest.main()
