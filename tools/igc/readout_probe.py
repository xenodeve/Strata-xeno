"""Read-only decomposition for bounded exploratory frozen-head diagnostics."""
import torch


def forward_parts(model,states,valid,positions,query):
    if (states.ndim!=4 or states.shape[1:]!=(80,4,2560) or valid.shape!=states.shape[:2]
            or valid.dtype!=torch.bool or positions.shape!=valid.shape or positions.dtype!=torch.long
            or query.shape!=(len(states),2560) or not bool(valid.any(1).all())):
        raise ValueError('bounded diagnostic geometry')
    safe=states.masked_fill(~valid[:,:,None,None],0)
    tokens=model.stream(model.norm(safe)).flatten(2)+model.position(positions)
    if model.encoder is not None:tokens=model.encoder(tokens,src_key_padding_mask=~valid)
    queries=model.queries[None,:,:].expand(len(states),-1,-1)
    if model.context is not None:queries=queries+model.context(query)[:,None,:]
    scores=torch.bmm(queries,model.key(tokens).transpose(1,2))/8
    attention=torch.softmax(scores.masked_fill(~valid[:,None,:],float('-inf')),dim=-1)
    values=model.value(tokens)
    return dict(logits=repool(model,values,queries,attention),attention=attention,values=values,queries=queries)


def repool(model,values,queries,attention):
    if (values.ndim!=3 or values.shape[2]!=64 or queries.shape!=(len(values),9,64)
            or attention.shape!=(len(values),9,values.shape[1])
            or not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(queries).all())
            or not bool(torch.isfinite(attention).all()) or bool((attention<0).any())
            or not torch.allclose(attention.sum(-1),torch.ones_like(attention.sum(-1)),rtol=0,atol=2e-6)):
        raise ValueError('finite normalized diagnostic pooling distribution')
    result=torch.bmm(attention,values)+queries
    return torch.cat([head(result[:,i]) for i,head in enumerate(model.heads)],dim=1)
