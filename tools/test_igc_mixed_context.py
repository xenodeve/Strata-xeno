"""Mixed source locators must retain row/mask order without whole-cache loads."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from tools.igc.context_cache import NpyRows
from tools.igc.mixed_context import MixedContextInputs


def row(identifier,split='train'):
    return dict(id=identifier,split=split,lhs=3,rhs=8,ready_op=1)


class MixedContext(unittest.TestCase):
    def fixture(self):
        base=np.arange(80*4*2560,dtype=np.float32).reshape(80,4,2560)
        sources={};members=[]
        for n,name in enumerate(('old','matched')):
            states=np.stack([base+(n*3+i)*1000000 for i in range(3)])
            valid=np.zeros((3,80),dtype=np.bool_)
            for i in range(3):valid[i,-i-2-n:]=True
            rows=[row(f'{name}-{i}') for i in range(3)];sources[name]=(states,valid,rows)
            members += [dict(source=name,index=i,id=r['id']) for i,r in enumerate(rows)]
        return sources,members

    def test_file_backed_order_duplicate_indices_masks_hc_and_channels(self):
        sources,members=self.fixture()
        with tempfile.TemporaryDirectory() as root:
            readers=[]
            try:
                for name,(values,valid,rows) in list(sources.items()):
                    writer=NpyRows.create(Path(root)/f'{name}.npy',3)
                    for i,v in enumerate(values):writer[i]=v
                    writer.close();reader=NpyRows(Path(root)/f'{name}.npy');readers.append(reader)
                    sources[name]=(reader,valid,rows)
                mixed=MixedContextInputs(sources,members,'cpu')
                select=np.array([5,0,4,5,1],dtype=np.int64);actual=mixed.batch(select)
                for i,member in enumerate(select):
                    ref=members[member];table,valid,_=sources[ref['source']]
                    expected=table[np.array([ref['index']],dtype=np.int64)][0]
                    np.testing.assert_array_equal(actual[0][i].numpy(),expected)
                    np.testing.assert_array_equal(actual[1][i].numpy(),valid[ref['index']])
                self.assertTrue(torch.equal(actual[0][0],actual[0][3]))
                self.assertFalse(torch.equal(actual[1][0],actual[1][1]))
                np.testing.assert_array_equal(actual[2].numpy(),np.tile(np.arange(80),(5,1)))
                self.assertFalse(bool(actual[3].any()))
            finally:
                for reader in readers:reader.close()

    def test_resolve_refuses_evaluation_rows_wrong_ids_and_out_of_range(self):
        sources,members=self.fixture()
        for bad in [{**members[0],'id':'wrong'},{**members[0],'index':99},{**members[0],'source':'other'}]:
            with self.assertRaises(ValueError):MixedContextInputs(sources,[bad],'cpu')
        sources['old'][2][0]['split']='development'
        with self.assertRaises(ValueError):MixedContextInputs(sources,members,'cpu')

    def test_numpy_fallback_requests_only_bounded_selected_rows(self):
        sources,members=self.fixture()
        class Bounded:
            def __init__(self,a):self.a=a;self.shape=a.shape;self.dtype=a.dtype;self.ndim=a.ndim;self.calls=[]
            def __len__(self):return len(self.a)
            def __getitem__(self,ix):self.calls.append(ix.copy());return self.a[ix]
            def __array__(self,*args):raise AssertionError('whole cache preload')
        wrappers=[]
        for name,(a,v,r) in list(sources.items()):
            w=Bounded(a);wrappers.append(w);sources[name]=(w,v,r)
        mixed=MixedContextInputs(sources,members,'cpu');mixed.batch(np.array([5,0,5],dtype=np.int64))
        self.assertTrue(all(len(ix)<=3 for w in wrappers for ix in w.calls))
        for bad in [np.array([],dtype=np.int64),np.array([6]),np.array([0.0]),np.zeros(65,dtype=np.int64)]:
            with self.assertRaises(ValueError):mixed.batch(bad)
        target=np.empty((3,80,4,2560),dtype=np.float32)
        self.assertIs(mixed.cpu_batch(np.array([5,0,5]),target)[0],target)
        with self.assertRaises(ValueError):mixed.cpu_batch(np.array([0]),target)

    def test_read_into_fills_callers_buffer_and_refuses_wrong_geometry(self):
        with tempfile.TemporaryDirectory() as root:
            w=NpyRows.create(Path(root)/'a.npy',2)
            try:
                a=np.ones((80,4,2560),dtype=np.float32);w[0]=a;w[1]=a*7
                out=np.empty((3,80,4,2560),dtype=np.float32)
                self.assertIs(w.read_into(np.array([1,0,1],dtype=np.int64),out),out)
                self.assertTrue(np.all(out[0]==7) and np.all(out[1]==1) and np.all(out[2]==7))
                with self.assertRaises(ValueError):w.read_into(np.array([0]),out)
            finally:w.close()


if __name__=='__main__':unittest.main()
