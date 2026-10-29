"""Read-only replay of the explicitly inspected development JSON evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

FILES = {
    'igc-native-context-cache-01/corpus.json': '0aa4fa0c34e45b47f213279a638d7d11d1a04f0256877ddcf5e8be6c803452d9',
    'igc-native-context-fit-02/predictions.json': '37c8200140c8a00461cab988f810fed5af4d33bffd5a2ce90f3e5bf94c7d386f',
    'igc-native-context-fit-02/report.json': '6b47259d3b8ce9fb88319cebced1fbbb6ce1d9dd188bfc6d069c707668133870',
    'igc-pipeline-replay-01/reference.json': 'f072e19dedad9b32136334c687a99c8006be23aa6ec08aa5f2946b7e5636bffc',
    'igc-output-fit-01/report.json': '8da2cf607098f4c4daed86d4f9216436d640ff9f1346a4bb2721f845751a2fd6',
    'igc-readiness-fit-01/report.json': '08989d4c4c07963d264a71570e6f79fa6fcc348291bd74993235adb7e22c0610',
}


def predicted_padding(raw):
    sizes = (4, 2, 2, 2, 2, 10, 10, 10, 10)
    if len(raw) != 9 or any(type(x) is not int or not 0 <= x < n for x, n in zip(raw, sizes)):
        raise ValueError('bounded categorical prediction')
    value = list(raw)
    if value[3] == 0:
        value[6] = 0
    if value[4] == 0:
        value[8] = 0
    return value


def fields(row):
    a, b = row['lhs'], row['rhs']
    return [row['ready_op'], int(a < 0), int(b < 0), int(abs(a) >= 10), int(abs(b) >= 10),
            abs(a) % 10, abs(a) // 10, abs(b) % 10, abs(b) // 10]


def summarize(rows, predictions):
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or len(rows) != len(predictions):
        raise ValueError('prediction identity/count')
    splits = defaultdict(list)
    for row in rows:
        p = by_id[row['id']]
        if p['target'] != fields(row) or len(p['predicted']) != 9:
            raise ValueError('canonical target/shape')
        if row['ready_op']:
            splits[row['split']].append((row, p['target'], p['predicted']))
    result = {}
    for split, selected in splits.items():
        confusion = defaultdict(Counter)
        lengths = defaultdict(lambda: [0, 0])
        values = defaultdict(lambda: [0, 0])
        associations = dict(ones_errors=0, padding_errors=0,
                            ones_matches_other_active_digit=0, padding_matches_own_ones=0)
        composition = Counter()
        for row, target, predicted in selected:
            composed = predicted_padding(predicted)
            composition['raw_exact'] += predicted == target
            composition['variant_exact'] += composed == target
            composition['tuple_recovered'] += predicted != target and composed == target
            length_key = f"{len(str(abs(row['lhs'])))}/{len(str(abs(row['rhs'])))}"
            lengths[length_key][0] += target == predicted
            lengths[length_key][1] += 1
            for side, indices in [('lhs', (1, 3, 5, 6)), ('rhs', (2, 4, 7, 8))]:
                sign, length, ones, tens = indices
                width = len(str(abs(row[side])))
                values[str(abs(row[side]))][0] += all(target[i] == predicted[i] for i in indices)
                values[str(abs(row[side]))][1] += 1
                active_bad = predicted[ones] != target[ones] or (target[length] == 1 and predicted[tens] != target[tens])
                padding_bad = target[length] == 0 and predicted[tens] != 0
                other_bad = any(predicted[j] != target[j] for j in (0, sign, length))
                if any(target[j] != predicted[j] for j in indices):
                    category = 'other' if other_bad else 'both' if active_bad and padding_bad else 'active_only' if active_bad else 'padding_only'
                    composition['operand_' + category] += 1
                for slot, index in [('ones', ones), ('tens', tens)]:
                    confusion[f'{side}/{width}/{slot}'][f'{target[index]}->{predicted[index]}'] += 1
                    if target[index] != predicted[index]:
                        if slot == 'ones':
                            other = 'rhs' if side == 'lhs' else 'lhs'
                            active = [abs(row[other]) % 10]
                            if abs(row[other]) >= 10:
                                active.append(abs(row[other]) // 10)
                            associations['ones_errors'] += 1
                            associations['ones_matches_other_active_digit'] += predicted[index] in active
                        elif width == 1:
                            associations['padding_errors'] += 1
                            associations['padding_matches_own_ones'] += predicted[index] == target[ones]
        result[split] = dict(positives=len(selected), exact=sum(t == p for _, t, p in selected),
            exact_by_length=dict(lengths), digit_confusion={k: dict(v) for k, v in confusion.items()},
            operand_exact_by_magnitude=dict(values), error_associations=associations,
            predicted_length_padding_variant=dict(composition))
    return result


def replay_counterfactual(records):
    counts = Counter()
    canonical = lambda s, n, one, ten: not (n == 0 and ten != 0) and not (n == 1 and ten == 0) and not (s and n == 0 and one == 0)
    for row in records:
        positive = bool(row['original_label']['ready_op'])
        target = fields(row['original_label'])
        secondary = row['secondary']
        selected = bool(row['ready']) and secondary[0] > 0 and canonical(secondary[1], secondary[3], secondary[5], secondary[6]) and canonical(secondary[2], secondary[4], secondary[7], secondary[8])
        counts[('positive' if positive else 'negative') + ('_selected' if selected else '_no_call')] += 1
        if positive:
            counts['old_selected'] += row['selection'] == 0
            counts['primary_exact'] += row['primary'] == target
            counts['secondary_exact'] += secondary == target
            counts['readiness_off_primary_wrong'] += not row['ready'] and row['primary'] != target
            counts['readiness_off_both_correct'] += not row['ready'] and row['primary'] == secondary == target
        counts['selected_wrong_tuple'] += selected and secondary != target
        counts['rescued_disagreements'] += selected and row['selection'] == 4
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    loaded = {}
    for name, expected in FILES.items():
        path = args.evidence_root / name
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError('bounded evidence file')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('evidence identity: ' + name)
        loaded[name] = json.loads(raw)
    corpus = loaded['igc-native-context-cache-01/corpus.json']
    fit = loaded['igc-native-context-fit-02/report.json']
    groups = {s: {(r['lhs'], r['rhs']) for r in corpus['rows'] if r['split'] == s and r['ready_op']}
              for s in ('new-values-seen-templates', 'new-values-new-templates')}
    output = loaded['igc-output-fit-01/report.json']
    readiness = loaded['igc-readiness-fit-01/report.json']
    value = dict(evidence_sha256=FILES, activation_stage=fit['activation_stage'], state_sha256=fit['trained_state_sha256'],
        input_metrics={name: summary['metrics'] for name, summary in fit['summaries'].items()},
        splits=summarize(corpus['rows'], loaded['igc-native-context-fit-02/predictions.json']),
        novel_axis_signed_pair_overlap=len(set.intersection(*groups.values())),
        first_attention_only=replay_counterfactual(loaded['igc-pipeline-replay-01/reference.json']['records']),
        output_mapper=dict(parameters=output['parameters'], controls=output['controls']),
        readiness=readiness['development'], scope='old inspected development; descriptive offline counterfactual, no policy change')
    agreement = Counter()
    for row in loaded['igc-pipeline-replay-01/reference.json']['records']:
        a, b = predicted_padding(row['primary']), predicted_padding(row['secondary'])
        agreement['primary_changed'] += a != row['primary']
        agreement['secondary_changed'] += b != row['secondary']
        agreement['agreement_raw'] += row['primary'] == row['secondary']
        agreement['agreement_variant'] += a == b
        agreement['ready_agreed_raw'] += bool(row['ready']) and row['primary'] == row['secondary'] and row['primary'][0] > 0
        agreement['ready_agreed_variant'] += bool(row['ready']) and a == b and a[0] > 0
    value['old64_padding_variant_agreement'] = dict(agreement)
    with args.output.open('x', encoding='utf-8') as target:
        json.dump(value, target, indent=2, ensure_ascii=False)
        target.write('\n')
    print(json.dumps(dict(complete=True, activation_stage=value['activation_stage'],
                         first_attention_only=value['first_attention_only'])))


if __name__ == '__main__':
    main()
