"""Offline checks of component benchmark accounting, never timed arithmetic."""
import copy
import unittest

from tools.igc.component_benchmark import normalize, validate_results


class ComponentAccounting(unittest.TestCase):
    def test_native_response_includes_only_the_empty_thinking_postlude(self):
        raw='\n<tool_response>\n391\n</tool_response>\n<think>\n\n</think>\n\n'
        self.assertEqual(normalize(raw,'qwenputer'),'391')
        for invalid in [raw+'junk',raw.replace('\n\n</think>','invented</think>'),raw+raw]:
            with self.assertRaises(ValueError):
                normalize(invalid,'qwenputer')

    def samples(self):
        result = []
        for block in range(2):
            for slot, arm in enumerate(['qwenputer', 'exact', 'exact', 'qwenputer']):
                result.append(dict(kind='sample', case=0, block=block, slot=slot, arm=arm,
                                   wall_ms=1.0 + block, status=0, actual='391',
                                   prepare_ms=0.1, issue_ms=0.2, wait_ms=0.7 + block,
                                   steps=100 if arm == 'qwenputer' else 0))
        return result

    def test_all_arms_checked_before_summary_and_no_oracle_time_added(self):
        report = validate_results(self.samples(), [{'expected': '391'}], 2)
        self.assertEqual(report['cases'][0]['exact']['n'], 4)
        self.assertEqual(report['cases'][0]['qwenputer']['median_ms'], 1.5)
        self.assertEqual(report['cases'][0]['exact']['p90_ms'], 2.0)
        self.assertEqual(report['cases'][0]['qwenputer']['steps'], [100] * 4)

    def test_wrong_answer_or_terminal_status_never_counts_as_speedup(self):
        for key, value in [('actual', '390'), ('status', 1)]:
            data = self.samples()
            data[0][key] = value
            with self.assertRaises(ValueError):
                validate_results(data, [{'expected': '391'}], 2)

    def test_missing_duplicate_or_reordered_slot_is_rejected(self):
        for change in ['missing', 'whole_block', 'duplicate', 'order']:
            data = copy.deepcopy(self.samples())
            if change == 'missing':
                data.pop()
            elif change == 'whole_block':
                data=data[:4]
            elif change == 'duplicate':
                data.append(data[0])
            else:
                data[0]['arm'] = 'exact'
            with self.assertRaises(ValueError):
                validate_results(data, [{'expected': '391'}], 2)

    def test_invalid_timing_and_inconsistent_stage_budget_are_rejected(self):
        for key, value in [('wall_ms', float('nan')), ('prepare_ms', -1), ('wait_ms', 20)]:
            data = self.samples()
            data[0][key] = value
            with self.assertRaises(ValueError):
                validate_results(data, [{'expected': '391'}], 2)


if __name__ == '__main__':
    unittest.main()
