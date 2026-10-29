"""Warm-start copies bounded tensors only; no optimizer state or silent casts."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from safetensors.torch import save
from tools.igc.initial_state import initialize,initialization_options,fresh_optimizer
from tools.igc.training_guards import state_fingerprint,verify_reset


class InitialState(unittest.TestCase):
    def artifact(self,root,tensors):
        path=Path(root)/'input.safetensors';path.write_bytes(save(tensors))
        return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_default_is_a_noop_and_requires_both_explicit_options(self):
        model=torch.nn.Linear(2,1);before=state_fingerprint(model)
        self.assertEqual(initialize(model),{'kind':'fresh_seeded'})
        self.assertEqual(state_fingerprint(model),before)
        for path,digest in [(Path('missing'),None),(None,'a'*64),(Path('x'),'bad')]:
            with self.assertRaises(ValueError):initialization_options(path,digest)

    def test_copy_is_exact_source_immutable_and_optimizer_fresh(self):
        with tempfile.TemporaryDirectory() as root:
            source=torch.nn.Linear(2,1);model=torch.nn.Linear(2,1)
            path,digest=self.artifact(root,source.state_dict());before=path.read_bytes()
            binding=initialize(model,path,digest)
            self.assertEqual(state_fingerprint(model),state_fingerprint(source))
            self.assertEqual(binding['artifact_sha256'],digest)
            optimizer=fresh_optimizer(model)
            self.assertEqual(len(optimizer.state),0)
            self.assertEqual(optimizer.param_groups[0]['lr'],.001)
            self.assertEqual(optimizer.param_groups[0]['weight_decay'],.01)
            with torch.no_grad():model.weight.add_(1)
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse(torch.equal(source.weight,model.weight))

    def test_digest_or_tensor_contract_failure_never_changes_target(self):
        for defect in ['digest','extra','missing','shape','dtype','nan']:
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as root:
                model=torch.nn.Linear(2,1);before=state_fingerprint(model)
                values={k:v.detach().clone() for k,v in model.state_dict().items()}
                if defect=='extra':values['optimizer.exp_avg']=torch.zeros(1)
                if defect=='missing':del values['bias']
                if defect=='shape':values['weight']=torch.zeros(1,3)
                if defect=='dtype':values['weight']=values['weight'].double()
                if defect=='nan':values['weight'].fill_(float('nan'))
                path,digest=self.artifact(root,values)
                if defect=='digest':digest='0'*64
                with self.assertRaises(ValueError):initialize(model,path,digest)
                self.assertEqual(state_fingerprint(model),before)

    def test_source_bound_is_checked_before_deserializing(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'large';path.write_bytes(b'0'*(2*1024**2+1))
            with self.assertRaises(ValueError):initialize(torch.nn.Linear(2,1),path,'0'*64)

    def test_price_recipe_refuses_a_different_initial_artifact(self):
        recipe={'initialization':{'kind':'copied_safetensors','artifact_sha256':'a'*64}}
        price=dict(success=True,optimizer_steps=0,backward_batches=1,initial_state_sha256='same',
            initial_batch_ids=['train'],source_sha256={},cache_identity={},protocol_sha256='p',recipe=recipe)
        verify_reset(price,'same',['train'],{},{},'p',recipe)
        with self.assertRaises(ValueError):verify_reset(price,'same',['train'],{},{},'p',{'initialization':{'kind':'fresh_seeded'}})

    def test_cli_forwards_explicit_initialization_pair(self):
        from tools.igc.embedding_readout import main
        with patch('tools.igc.embedding_readout.train') as train:
            main(['--dataset','cache','--sources','sources','--protocol','p','--output','new',
                '--initial-state','weights','--initial-state-sha256','a'*64])
        self.assertEqual(train.call_args.kwargs['initial_state'],Path('weights'))
        self.assertEqual(train.call_args.kwargs['initial_state_sha256'],'a'*64)


if __name__=='__main__':unittest.main()
