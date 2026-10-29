import unittest
from tools.igc.output_corpus import dataset
from tools.igc.balanced_corpus import expanded_dataset
from tools.igc.mapper_corpus import evaluation_manifest


class OutputCorpus(unittest.TestCase):
    def test_frozen_independent_results_splits_and_context_labels(self):
        data=dataset();self.assertEqual(data,dataset())
        rows=data['rows'];self.assertEqual(len(rows),30)
        fit=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='development']
        self.assertEqual((len(fit),len(dev)),(24,6))
        self.assertEqual({r['operation'] for r in rows},{1,2,3})
        self.assertFalse({r['answer'] for r in fit}&{r['answer'] for r in dev})
        reserved={(r['lhs'],r['rhs']) for r in expanded_dataset()['rows']+evaluation_manifest()['rows']}
        self.assertFalse(reserved&{(r['lhs'],r['rhs']) for r in rows})
        digits=set()
        for row in rows:
            a,b,op=row['lhs'],row['rhs'],row['operation']
            self.assertEqual(row['answer'],{1:lambda:a+b,2:lambda:a-b,3:lambda:a*b}[op]())
            self.assertIn(str(row['answer']),row['completion'])
            self.assertIn('</think>',row['completion'])
            if row['split']=='train':digits|=set(str(row['answer']))&set('0123456789')
        self.assertEqual(digits,set('0123456789'))


if __name__=='__main__':unittest.main()
