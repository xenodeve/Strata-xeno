"""Preflight and explicitly gated offline firmware bootstrap, never serving code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from tools.qwenputer.tokenizer import NativeInputError, NativeText
from tools.qwenputer.firmware import parse_elf, read_firmware
from tools.qwenputer.source import MANIFEST_SHA256, verify_source


def preflight(source_root: Path, tokenizer_dir: Path, firmware_path: Path,
              max_boot_tokens: int = 200000) -> dict:
    if type(max_boot_tokens) is not int or not 1 <= max_boot_tokens <= 1000000:
        raise NativeInputError('boot_token_limit')
    source = verify_source(source_root)
    text = NativeText.from_pack(tokenizer_dir)
    data = read_firmware(firmware_path)
    firmware = parse_elf(data)
    return {'source_revision': source['revision'], 'source_manifest_sha256': MANIFEST_SHA256,
            'firmware_sha256': firmware.sha256, 'firmware_bytes': len(data),
            'entry': firmware.entry, 'vocab_size': text.n,
            'tokenizer_binding_sha256': text.binding_sha256,
            'max_boot_tokens': max_boot_tokens, 'would_execute': False}


def capture_boot(machine, trace, limit):
    if not machine.halted or not 1 <= len(trace) <= limit:
        raise NativeInputError('boot_limit')
    if machine.fault is not None or machine.exit_code != 0:
        raise NativeInputError('boot_fault')
    if machine.out:
        raise NativeInputError('boot_output')
    groups = [(machine.x, 32), (machine.ram, 8192), (machine.rom, 65536)]
    if any(len(values) != count or any(type(v) is not int or not 0 <= v <= 0xffffffff
                                      for v in values) for values, count in groups):
        raise NativeInputError('boot_state')
    if (not 0 <= machine.anchor < 65536 or not 0 <= machine.n <= 63
            or machine.anchor + machine.n >= 65536 or len(machine.drom) > 131072
            or len(machine.bn) != 8 or any(len(row) != 128 or any(type(v) is not int or not 0 <= v <= 9
                                                                 for v in row) for row in machine.bn)):
        raise NativeInputError('boot_state')
    return {'anchor': machine.anchor, 'n': machine.n, 'registers': list(machine.x),
            'ram': list(machine.ram), 'rom': list(machine.rom), 'drom_hex': machine.drom.hex(),
            'digits': [list(row) for row in machine.bn], 'resume_after_park': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[2] / 'third_party/qwenputer')
    parser.add_argument('--tokenizer-dir', type=Path, required=True)
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-boot-tokens', type=int, default=200000)
    parser.add_argument('--execute-reviewed-reference', action='store_true',
                        help='requires action-time approval for the pinned compiler/firmware operation')
    args = parser.parse_args(argv)
    report = preflight(args.source_root, args.tokenizer_dir, args.firmware, args.max_boot_tokens)
    if not args.execute_reviewed_reference:
        print(json.dumps(report, sort_keys=True))
        return 0
    if args.output is None or args.output.exists():
        raise NativeInputError('bootstrap_output')
    if any(name == 'chipformer' or name.startswith('chipformer.') for name in sys.modules):
        raise NativeInputError('compiler_already_loaded')
    # This branch is deliberately separate from data-only preflight. No upstream
    # module or firmware has been executed merely by importing this tool.
    args.output.mkdir(parents=True, exist_ok=False)
    firmware = read_firmware(args.firmware)
    if hashlib.sha256(firmware).hexdigest() != report['firmware_sha256']:
        raise NativeInputError('bootstrap_input_changed')
    frozen_elf = args.output / 'validated-firmware.elf'
    frozen_elf.write_bytes(firmware)
    text = NativeText.from_pack(args.tokenizer_dir)
    if text.binding_sha256 != report['tokenizer_binding_sha256']:
        raise NativeInputError('bootstrap_input_changed')
    verify_source(args.source_root)
    previous = list(sys.path)
    sys.path.insert(0, str(args.source_root.resolve() / 'experimental/riscv'))
    try:
        from chipformer.rv import core, elf, ref
        circuit = core.build(text=text)
        machine = ref.Machine.from_program(elf.program(frozen_elf))
        trace = machine.run(max_records=args.max_boot_tokens)
        state = capture_boot(machine, trace, args.max_boot_tokens)
        core.snapshot(circuit, machine)
        report.update(would_execute=True, kind='offline-reference-boot', version=1,
                      native_runtime=False, stats=circuit.stats(), boot_tokens=len(trace),
                      instructions=machine.steps,
                      initialized_tables=sum(getattr(node, 'init', None) is not None for node in circuit.nodes))
    finally:
        sys.path[:] = previous
    for name, value in [('boot-state.json', state), ('boot-trace.json', trace)]:
        data = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        (args.output / name).write_bytes(data)
        report[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
