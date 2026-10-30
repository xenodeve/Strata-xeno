"""One preallocated contextual batch gathered from verified cache row locators."""
import numpy as np
import torch
from tools.igc.context_cache import ContextInputs,NpyRows


class MixedContextInputs:
    def __init__(self,sources,members,device):
        if set(sources)!={'old','matched'} or not 1<=len(members)<=11264:
            raise ValueError('bounded two-source replay')
        for states,valid,rows in sources.values():
            ContextInputs(states,valid,'cpu')
            if len(rows)!=len(states):raise ValueError('source metadata row count')
        self.sources=sources;self.members=tuple(dict(m) for m in members);self.device=device;self.rows=[]
        for m in self.members:
            if (set(m)!={'source','index','id'} or m['source'] not in sources or type(m['index']) is not int
                    or not 0<=m['index']<len(sources[m['source']][0])):
                raise ValueError('replay source row locator')
            r=sources[m['source']][2][m['index']]
            if r['id']!=m['id'] or r['split']!='train':raise ValueError('replay ID/TRAIN identity')
            self.rows.append(r)
        if len({(m['source'],m['id']) for m in self.members})!=len(self.members):raise ValueError('duplicate replay member')

    def cpu_batch(self,selection,destination=None):
        if (not isinstance(selection,np.ndarray) or selection.ndim!=1 or not 1<=len(selection)<=64
                or selection.dtype not in (np.int32,np.int64) or np.any(selection<0) or np.any(selection>=len(self.members))):
            raise ValueError('bounded replay member selection')
        values=(np.empty((len(selection),80,4,2560),dtype=np.float32) if destination is None else destination)
        if (not isinstance(values,np.ndarray) or values.shape!=(len(selection),80,4,2560)
                or values.dtype!=np.float32 or not values.flags.c_contiguous or not values.flags.writeable):
            raise ValueError('bounded replay destination')
        mask=np.empty((len(selection),80),dtype=np.bool_)
        refs=[self.members[i] for i in selection];start=0
        while start<len(refs):
            name=refs[start]['source'];end=start+1
            while end<len(refs) and refs[end]['source']==name:end+=1
            indices=np.array([r['index'] for r in refs[start:end]],dtype=np.int64)
            states,valid,_=self.sources[name]
            if isinstance(states,NpyRows):states.read_into(indices,values[start:end])
            else:values[start:end]=states[indices]
            mask[start:end]=valid[indices];start=end
        if not np.isfinite(values).all():raise ValueError('nonfinite replay batch')
        positions=np.broadcast_to(np.arange(80,dtype=np.int64),mask.shape).copy()
        return values,mask,positions,np.zeros((len(selection),2560),dtype=np.float32)

    def batch(self,selection):
        return tuple(torch.from_numpy(a).to(self.device) for a in self.cpu_batch(selection))
