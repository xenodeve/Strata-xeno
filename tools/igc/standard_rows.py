"""The prompts of the published few-shot protocols, as rows for external_bench --rows. MATH: the four Minerva exemplars of Lewkowycz et al. 2022.
BBH: the three CoT exemplars of Suzgun et al. 2022. MMLU-Pro and SuperGPQA: their own five exemplars, in the form of each suite's evaluator.
standard_rows.py <data dir> <output dir>: the data dir holds what was fetched, read as data.
standard_rows.py plan <data dir> <output dir>: the seeded rows of the full test plan, for each source the data dir holds."""
import ast, json, random, re, sys, textwrap
from collections import Counter
from pathlib import Path
from tools.igc.external_score import question_id
from tools.igc.standard_score import last_boxed, normalize_math


def minerva_exemplars(path):
    """The list inside list_fewshot_samples, parsed and not imported: the file is downloaded."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    body = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'list_fewshot_samples').body
    return ast.literal_eval(next(n for n in body if isinstance(n, ast.Return)).value)


def math_rows(data):
    shots = ''.join(f"Problem:\n{e['problem']}\n\nSolution: {e['solution']}\n\n" for e in minerva_exemplars(data / 'minerva_utils.py'))
    for row in json.loads((data / 'math.json').read_text(encoding='utf-8')):
        gold = last_boxed(row['solution'])
        yield dict(id=question_id('stdmath', row['problem']), suite='stdmath', prompt=f"{shots}Problem:\n{row['problem']}\n\nSolution:",
                   answer=normalize_math(gold or ''), limit=2048, group=row['config'], level=row.get('level'))


def bbh_rows(data):
    for task, entry in json.loads((data / 'bbh.json').read_text(encoding='utf-8')).items():
        shots = entry['prompt'].split('-----')[-1].strip()
        for row in entry['examples']:
            yield dict(id=question_id('bbh', task + row['input']), suite='bbh', prompt=f"{shots}\n\nQ: {row['input']}\nA: Let's think step by step.",
                       answer=row['target'], limit=1024, group=task)


LETTERS = 'ABCDEFGHIJKLMNOP'
LETTER_TAIL = '\n\nFinish your answer with "the answer is (X)" where X is the correct letter.'


HLE_TAIL = ('\n\nYour response should be in the following format:\nExplanation: {your explanation for your final answer}\n'
            'Exact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}')


def hle_rows(path, categories=('Math', 'Physics', 'Computer Science/AI'), sample=30, seed=1):
    """Humanity's Last Exam, read from a local parquet that stays on the test machine: its authors ask that the questions are not shared.
    Text only, numeric gold only: those are the rows a program can reach and a rule can score with no judge."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=['id', 'question', 'image', 'answer', 'answer_type', 'category']).to_pylist()
    keep = [r for r in table if not r['image'] and r['answer_type'] == 'exactMatch' and r['category'] in categories
            and re.fullmatch(r'-?\d+(?:\.\d+)?(?:/\d+)?', r['answer'].replace(',', '').strip())]
    random.Random(seed).shuffle(keep)
    return [dict(id='hle-' + r['id'][:12], suite='hle', prompt=r['question'] + HLE_TAIL, answer=r['answer'].strip(), limit=8192, group=r['category']) for r in keep[:sample]], len(keep)


def plain_prompt(row):
    """The question alone, with no exemplars, for the arms with the thinking on."""
    text = row['prompt']
    if row['suite'] == 'stdmath':
        return text.rsplit('Problem:\n', 1)[1].rsplit('\n\nSolution:', 1)[0]
    if row['suite'] == 'mmlupro':
        return text.rsplit('Question:\n', 1)[1].rsplit("Answer: Let's think", 1)[0].strip() + LETTER_TAIL
    if row['suite'] == 'supergpqa':
        return text.rsplit('Question: \n', 1)[1].rsplit("Answer: Let's think", 1)[0].strip() + LETTER_TAIL
    return text


def mmlu_pro_rows(data):
    head = (data / 'mp-init.txt').read_text(encoding='utf-8').strip() + '\n\n'
    def body(row):
        return 'Question:\n' + row['question'] + '\nOptions:\n' + ''.join(f'{LETTERS[i]}. {o}\n' for i, o in enumerate(row['options']))
    shots = {}
    for row in json.loads((data / 'mp-val.json').read_text(encoding='utf-8')):
        text = body(row) + row['cot_content'].replace("A: Let's think step by step.", "Answer: Let's think step by step.") + '\n\n'
        shots.setdefault(row['category'], []).append(text)
    for row in json.loads((data / 'mp-test.json').read_text(encoding='utf-8')):
        cat = row['category']
        yield dict(id=question_id('mmlupro', row['question']), suite='mmlupro', prompt=head.replace('{$}', cat) + ''.join(shots[cat][:5]) + body(row) + "Answer: Let's think step by step.",
                   answer=row['answer'], limit=1024, group=cat)


def super_gpqa_rows(data, sample=2000):
    """A seeded sample by discipline: the whole suite is 26,529 questions, and the sample shows the trend before it is run. sample=None: all of it."""
    text = (data / 'sg-five.yaml').read_text(encoding='utf-8')
    prompt = textwrap.dedent(text[text.index('- |') + 3:].split('\n', 1)[1]).strip()
    rows = json.loads((data / 'sg.json').read_text(encoding='utf-8'))
    random.Random(1).shuffle(rows)
    per = {}
    for row in rows:
        per.setdefault(row['discipline'], []).append(row)
    keep = rows if sample is None else [r for group in per.values() for r in group[:round(sample * len(group) / len(rows))]]
    random.Random(2).shuffle(keep)
    for row in keep:
        question = row['question'] + '\n' + ''.join(f'{LETTERS[i]}) {o}\n' for i, o in enumerate(row['options']))
        yield dict(id=question_id('supergpqa', row['uuid']), suite='supergpqa', prompt=prompt.replace('{{', '{').replace('}}', '}').replace('{}', question.strip()),
                   answer=row['answer_letter'], limit=1024, group=row['discipline'], field=row['field'],
                   calculation=str(row['is_calculation']) == 'True', difficulty=row['difficulty'])


def gsm_rows(path):
    """GSM8K's test split, in the bench's own form (external_bench.cases)."""
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        yield dict(id=question_id('gsm8k', row['question']), suite='gsm8k', prompt=row['question'],
                   answer=int(row['answer'].split('####')[1].strip().replace(',', '')), limit=1536)


def gpqa_rows(path, domains=('Physics', 'Chemistry')):
    """GPQA Diamond, read from a local csv that stays on the test machine: its authors ask that the questions are not shared.
    The options are shuffled by the row's own Record ID, so a row holds the same letters in every arm and on every machine."""
    import csv
    with open(path, encoding='utf-8', newline='') as f:
        table = list(csv.DictReader(f))
    for row in table:
        if row['High-level domain'] not in domains:
            continue
        options = [row['Correct Answer']] + [row[f'Incorrect Answer {i}'] for i in (1, 2, 3)]
        options = [o.strip() for o in options]
        order = list(range(4))
        random.Random(row['Record ID']).shuffle(order)
        text = row['Question'].strip() + '\n\n' + ''.join(f'{LETTERS[i]}) {options[k]}\n' for i, k in enumerate(order))
        yield dict(id='gpqa-' + row['Record ID'], suite='gpqa', prompt=text.rstrip() + LETTER_TAIL, answer=LETTERS[order.index(0)],
                   limit=8192, group=row['High-level domain'])


def pick(rows, n, keep=lambda r: True, seed=1):
    """n rows that pass `keep`, by a seeded shuffle: the same rows on every machine."""
    rows = [r for r in rows if keep(r)]
    random.Random(seed).shuffle(rows)
    return rows[:n], len(rows)


# The full test plan (docs/plans/2026-11-02-full-test-plan.md): what each stage keeps. MIP computes, so each suite keeps the questions with a
# calculation in them where it can tell; the rest is said with the row count it left.
PLAN = {
    'gsm': ('std-gsm60.jsonl', 60, lambda r: True),
    'math5': ('std-math5.jsonl', 60, lambda r: r['level'] == 'Level 5'),
    'mmlupro': ('std-mmlupro-calc.jsonl', 60, lambda r: r['group'] in ('math', 'physics', 'engineering')),
    'supergpqa': ('std-supergpqa-calc.jsonl', 60, lambda r: r['calculation'] and (r['field'] in ('Mathematics', 'Physics') or r['group'] == 'Engineering')),
    'gpqa': ('std-gpqa60.jsonl', 60, lambda r: True),
    'lcb': ('std-lcb60.jsonl', 60, lambda r: True),
}


def plan_sources(data):
    """Each stage's rows, where its source is in the data dir: the sources are spread over this PC and the VMs."""
    found = {}
    if (data / 'gsm8k-test.jsonl').exists():
        found['gsm'] = lambda: gsm_rows(data / 'gsm8k-test.jsonl')
    if (data / 'math.json').exists():
        found['math5'] = lambda: math_rows(data)
    if (data / 'mp-test.json').exists():
        found['mmlupro'] = lambda: mmlu_pro_rows(data)
    if (data / 'sg.json').exists():
        found['supergpqa'] = lambda: super_gpqa_rows(data, sample=None)
    if (data / 'gpqa_diamond.csv').exists():
        found['gpqa'] = lambda: gpqa_rows(data / 'gpqa_diamond.csv')
    if (data / 'std-lcb.jsonl').exists():
        found['lcb'] = lambda: (json.loads(line) for line in (data / 'std-lcb.jsonl').read_text(encoding='utf-8').splitlines())
    return found


def build_plan(data, output):
    for stage, make in plan_sources(data).items():
        name, n, keep = PLAN[stage]
        rows, passed = pick(list(make()), n, keep)
        with open(output / name, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        print(stage, name, len(rows), 'of', passed, 'that pass the rule', dict(sorted(Counter(r.get('group') for r in rows).items())))


if __name__ == '__main__' and sys.argv[1] == 'plan':   # plan <data dir> <output dir>: the full test plan's rows from the sources found
    build_plan(Path(sys.argv[2]), Path(sys.argv[3]))
elif __name__ == '__main__':
    data, output = Path(sys.argv[1]), Path(sys.argv[2])
    for name, rows in (('std-math.jsonl', math_rows(data)), ('std-bbh.jsonl', bbh_rows(data)), ('std-mmlupro.jsonl', mmlu_pro_rows(data)), ('std-supergpqa.jsonl', super_gpqa_rows(data))):
        rows = list(rows)
        with open(output / name, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        print(name, len(rows), 'empty gold', sum(not r['answer'] for r in rows), re.sub(r'\s+', ' ', rows[0]['prompt'][-120:]))
