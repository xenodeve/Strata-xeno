"""Coverage and held-out axes for the next declared input-readout experiment."""
import unittest
from tools.igc.balanced_corpus import dataset,expanded_dataset,WITHHELD
from tools.igc.mapper_corpus import corpus,evaluation_manifest
from tools.igc.readout import targets


class BalancedCorpus(unittest.TestCase):
    def test_expanded_fit_preserves_old_manifest_and_development_pair_exclusion(self):
        old=dataset()
        self.assertEqual(old['rows_sha256'],'193f34b29fef6e884a4074a2e20db858f4467e2cc138365c76995d39ed8edf62')
        new=expanded_dataset()
        train=[r for r in new['rows'] if r['split']=='train']
        self.assertEqual(len(train),10240)
        self.assertEqual(len(new['rows']),10624)
        old_dev={(r['lhs'],r['rhs']) for r in old['rows'] if r['split']!='train'}
        self.assertFalse(old_dev&{(r['lhs'],r['rhs']) for r in new['rows']})
        self.assertGreater(len({r['template'] for r in train if r['ready_op']==0}),10)
        self.assertTrue(all(abs(r[k]) not in WITHHELD for r in train for k in ('lhs','rhs')))

    def test_stable_coverage_and_independent_development_axes(self):
        first=dataset();second=dataset()
        self.assertEqual(first,second)
        train=[r for r in first['rows'] if r['split']=='train']
        self.assertEqual(len(train),4096)
        self.assertTrue(all(abs(r[k]) not in WITHHELD for r in train for k in ('lhs','rhs')))
        labels=targets([r for r in train if r['ready_op']])
        for field in (5,6,7,8):self.assertEqual(set(labels[:,field]),set(range(10)))
        train_templates={r['template'] for r in train}
        for axis in ('seen-values-new-templates','new-values-seen-templates','new-values-new-templates'):
            rows=[r for r in first['rows'] if r['split']==axis]
            self.assertEqual(len(rows),128)
            self.assertEqual({r['ready_op'] for r in rows},set(range(4)))
            for r in rows:
                self.assertEqual(r['template'] in train_templates,axis=='new-values-seen-templates')
                self.assertEqual(any(abs(r[k]) in WITHHELD for k in ('lhs','rhs')),axis!='seen-values-new-templates')

    def test_pilot_and_untouched_evaluation_pairs_are_excluded(self):
        reserved={(r['lhs'],r['rhs']) for r in corpus()+evaluation_manifest()['rows']}
        observed={(r['lhs'],r['rhs']) for r in dataset()['rows']}
        self.assertFalse(observed&reserved)
        for r in dataset()['rows']:
            self.assertTrue(-99<=r['lhs']<=99 and -99<=r['rhs']<=99)
            self.assertIn(r['expression'],r['prompt'])


if __name__=='__main__':unittest.main()
