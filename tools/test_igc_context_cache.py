"""Context archive identity and bounded minibatches are not inferred from filenames."""
import struct
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from tools.igc.context_cache import decode_archive,ContextInputs,NpyRows


class ContextCache(unittest.TestCase):
    def test_file_selection_preserves_distinct_masks_streams_channels_and_duplicate_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'states.npy';writer=NpyRows.create(path,3)
            base=np.arange(80*4*2560,dtype=np.float32).reshape(80,4,2560)
            for row in range(3):writer[row]=base+row*1000000
            writer.flush();writer.close()
            reader=NpyRows(path)
            try:
                mask=np.zeros((3,80),dtype=np.bool_)
                for row in range(3):mask[row,-row-2:]=True
                selection=np.array([2,0,2],dtype=np.int64)
                actual,valid,positions,query=ContextInputs(reader,mask,'cpu').batch(selection)
                for i,row in enumerate(selection):
                    np.testing.assert_array_equal(actual[i].numpy(),base+row*1000000)
                np.testing.assert_array_equal(valid.numpy(),mask[selection])
                np.testing.assert_array_equal(positions.numpy(),np.tile(np.arange(80),(3,1)))
                self.assertFalse(torch.equal(valid[0],valid[1]));self.assertTrue(torch.equal(valid[0],valid[2]))
                self.assertFalse(bool(query.any()))
            finally:reader.close()

    def test_file_rows_use_bounded_io_and_refuse_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'states.npy'
            writer=NpyRows.create(path,2)
            expected=np.ones((80,4,2560),dtype=np.float32)
            writer[0]=expected;writer[1]=expected*3;writer.flush();writer.close()
            reader=NpyRows(path)
            self.assertFalse(isinstance(reader,np.memmap))
            np.testing.assert_array_equal(reader[np.array([1,0])][:,0,0,0],[3,1])
            reader.close()
            with path.open('r+b') as stream:stream.truncate(path.stat().st_size-1)
            with self.assertRaises(ValueError):NpyRows(path)
    def archive(self):
        tokens=np.arange(84,dtype=np.int32)
        raw=b'IGCATN01'+struct.pack('<IIIIIQQ',1,4,2560,80,80,83,84)
        for p in range(3,83):
            raw+=struct.pack('<qII',p,int(tokens[p]),1)+np.full((4,2560),p,dtype='<f4').tobytes()
        return tokens,raw

    def test_identity_and_left_window_are_preserved(self):
        tokens,raw=self.archive();values,valid=decode_archive(raw,tokens)
        self.assertEqual(values.shape,(80,4,2560));self.assertTrue(valid.all())
        np.testing.assert_array_equal(values[:,0,0],np.arange(3,83))

    def test_truncation_wrong_owner_tokens_and_nonfinite_are_refused(self):
        tokens,raw=self.archive()
        changed=bytearray(raw);struct.pack_into('<I',changed,44+8,99)
        nonfinite=bytearray(raw);struct.pack_into('<f',nonfinite,44+16,float('nan'))
        for invalid in (raw[:-1],raw+b'x',bytes(changed),bytes(nonfinite),b'BADMAGIC'+raw[8:]):
            with self.assertRaises(ValueError):decode_archive(invalid,tokens)

    def test_bounded_selection_returns_activations_and_position_not_row_ids(self):
        states=np.ones((2,80,4,2560),dtype=np.float32);states[1]*=3
        valid=np.ones((2,80),dtype=np.bool_);valid[:,0]=False
        cache=ContextInputs(states,valid,'cpu')
        values,mask,positions,query=cache.batch(np.array([1,0],dtype=np.int64))
        self.assertEqual(float(values[0,1,0,0]),3);self.assertEqual(float(values[1,1,0,0]),1)
        self.assertTrue(torch.equal(mask,torch.from_numpy(valid)))
        self.assertEqual(positions[0].tolist(),list(range(80)));self.assertFalse(bool(query.any()))
        for selection in (np.array([-1]),np.array([2]),np.zeros(65,dtype=np.int64),np.array([0.5])):
            with self.assertRaises(ValueError):cache.batch(selection)


if __name__=='__main__':unittest.main()
