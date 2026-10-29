import unittest
from tools.igc.output_observe import position_examples,validate_launcher_env


class Tokenizer:
    def encode(self,text):return list(text.encode('ascii'))
    def decode_bytes(self,ids):return bytes(ids)


class PositionExamples(unittest.TestCase):
    def test_launcher_gpu_mapping_is_required_before_model_start(self):
        validate_launcher_env({'CUDA_VISIBLE_DEVICES':'1,0'})
        validate_launcher_env({'CUDA_VISIBLE_DEVICES':'1,0','CUDA_DEVICE_ORDER':''})
        for env in ({},{'CUDA_VISIBLE_DEVICES':'0,1'},
                    {'CUDA_VISIBLE_DEVICES':'1,0','CUDA_DEVICE_ORDER':'PCI_BUS_ID'}):
            with self.assertRaises(ValueError):validate_launcher_env(env)
    def test_future_targets_never_enter_the_captured_input(self):
        rows=[dict(id='a',prompt='Q',completion='12E',split='train',answer=12)]
        result=position_examples(rows,Tokenizer(),b'S',b'A')
        self.assertEqual(len(result),3)
        self.assertEqual([r['ids'] for r in result],[[83,81,65],[83,81,65,49],[83,81,65,49,50]])
        self.assertEqual([r['target'] for r in result],[49,50,69])
        self.assertEqual([r['invocation_position'] for r in result],[2,2,2])
        self.assertEqual([r['position'] for r in result],[2,3,4])

    def test_bound_precedes_model_launch_and_duplicates_are_refused(self):
        row=dict(id='a',prompt='Q',completion='1',split='train',answer=1)
        with self.assertRaises(ValueError):position_examples([row,row],Tokenizer(),b'S',b'A')
        with self.assertRaises(ValueError):position_examples([{**row,'completion':'1'*1025}],Tokenizer(),b'S',b'A')


if __name__=='__main__':unittest.main()
