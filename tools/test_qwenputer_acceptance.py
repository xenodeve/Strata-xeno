"""A completed native program is distinct from a verified arithmetic answer."""
import unittest
from tools.qwenputer.acceptance import classify_result


class Text:
    printable = tuple(range(258))

    @staticmethod
    def decode_bytes(ids):
        return bytes(ids)


class Acceptance(unittest.TestCase):
    def result(self, body=b'391', **fields):
        return dict(kind='native-execution', evidence_level='observed-execution',
                    artifact_sha256='a' * 64, reason=3, error=0,
                    text_ids=[256, *body, 257, 10], **fields)

    def test_independent_exact_rational_checker(self):
        for body, expression in ((b'391', '17*23'), (b'1/2', '1/3+1/6'),
                                 (b'-3', '-7//3')):
            with self.subTest(expression=expression):
                result = classify_result(self.result(body), Text(), expression, 'a' * 64)
                self.assertTrue(result['passed'], result)
                self.assertEqual(result['outcome'], 'verified-selected-test')

    def test_wrong_answer_and_refusal_are_not_success(self):
        for body, outcome in ((b'392', 'wrong-answer'), (b'error', 'firmware-refusal')):
            result = classify_result(self.result(body), Text(), '17*23', 'a' * 64)
            self.assertFalse(result['passed'])
            self.assertEqual(result['outcome'], outcome)

    def test_failure_cannot_publish_partial_answer(self):
        for reason in (0, 1, 2, 4, 5, 6):
            result = self.result()
            result['reason'] = reason
            checked = classify_result(result, Text(), '17*23', 'a' * 64)
            self.assertFalse(checked['passed'])
            self.assertEqual(checked['outcome'], 'execution-incomplete')
            self.assertNotIn('body', checked)

    def test_identity_and_envelope_are_required(self):
        for key, value in (('artifact_sha256', 'b' * 64), ('kind', 'reference'),
                           ('text_ids', [*b'391']), ('text_ids', [256, *b'391', 257, 256]),
                           ('text_ids', [256, *b'391', 257, 65]),
                           ('text_ids', [256, *b'391', 257, *b'<think>wrong</think>']),
                           ('text_ids', [256, *b'391\x00', 257]), ('error', 16)):
            with self.subTest(key=key, value=value):
                result = self.result()
                result[key] = value
                self.assertFalse(classify_result(result, Text(), '17*23', 'a' * 64)['passed'])

    def test_pinned_firmware_empty_thinking_suffix_is_framing(self):
        result = self.result()
        result['text_ids'] += list(b'<think>\n\n</think>\n\n')
        self.assertTrue(classify_result(result, Text(), '17*23', 'a' * 64)['passed'])


if __name__ == '__main__':
    unittest.main()
