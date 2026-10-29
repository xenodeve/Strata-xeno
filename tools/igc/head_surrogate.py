"""Offline late-head STE feasibility; no decoder adapter or training claim.

Forward uses the engine's actual Q8_1 activation quantizer and Q8_0 head.
Backward uses W_dequant.T @ delta, an explicit straight-through estimator.
This is a late head-input variant, not backward through Gyro's frozen layers.
The process-per-call bridge is for feasibility and independent verification;
it is not the production execution path or a latency optimization.
"""
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch.autograd.function import once_differentiable


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, head):
        ctx.head = head
        return head._forward(value)

    @staticmethod
    @once_differentiable
    def backward(ctx, delta):
        return ctx.head._backward(delta), None


class FrozenHeadSTE:
    def __init__(self, probe, pack, model, output):
        self.probe, self.pack, self.model = map(Path, (probe, pack, model))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.calls = 0

    @staticmethod
    def _validate(value, count):
        if value.device.type != 'cpu' or value.dtype != torch.float32 or value.shape != (count,):
            raise ValueError('head surrogate requires a CPU float32 vector of the declared shape')
        if not torch.isfinite(value).all().item():
            raise ValueError('head surrogate requires finite input')

    def _run(self, value, backward):
        count = 248320 if backward else 2560
        self._validate(value, count)
        self.calls += 1
        run = self.output / f'call-{self.calls:04d}'
        run.mkdir(exist_ok=False)
        source, result = run / 'input.bin', run / 'result.bin'
        value.detach().contiguous().numpy().astype('<f4', copy=False).tofile(source)
        command = [str(self.probe), '--pack', str(self.pack), '--model', str(self.model)]
        if backward:
            command += ['--zero', '--vjp-input', str(source), '--vjp-out', str(result)]
        else:
            command += ['--mixed-input', str(source), '--logits-out', str(result)]
        # The task's outer headless launcher owns timeout cleanup for the whole
        # experiment. This executable starts no child processes of its own.
        with (run / 'stdout.log').open('w', encoding='utf-8') as stdout, \
                (run / 'stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=30,
                                     creationflags=0x08000000)
        if process.returncode:
            raise RuntimeError(f'native head probe failed ({process.returncode}); evidence: {run}')
        expected = 2560 if backward else 248320
        if result.stat().st_size != expected * 4:
            raise RuntimeError('native head result size mismatch')
        data = np.fromfile(result, dtype='<f4')
        if not np.isfinite(data).all():
            raise RuntimeError('native head result is nonfinite')
        return torch.from_numpy(data)

    def _forward(self, value):
        return self._run(value, False)

    def _backward(self, delta):
        return self._run(delta, True)

    def project(self, value):
        self._validate(value, 2560)
        return _Projection.apply(value, self)
