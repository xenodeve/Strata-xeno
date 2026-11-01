"""The prompts of the published few-shot protocols, as rows for external_bench --rows. MATH: the four Minerva exemplars of Lewkowycz et al. 2022.
BBH: the three CoT exemplars of Suzgun et al. 2022. MMLU-Pro and SuperGPQA: their own five exemplars, in the form of each suite's evaluator.
standard_rows.py <data dir> <output dir>: the data dir holds what was fetched, read as data."""
import ast, json, random, re, sys, textwrap
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
                   answer=normalize_math(gold or ''), limit=2048, group=row['config'])


def bbh_rows(data):
    for task, entry in json.loads((data / 'bbh.json').read_text(encoding='utf-8')).items():
        shots = entry['prompt'].split('-----')[-1].strip()
        for row in entry['examples']:
            yield dict(id=question_id('bbh', task + row['input']), suite='bbh', prompt=f"{shots}\n\nQ: {row['input']}\nA: Let's think step by step.",
                       answer=row['target'], limit=1024, group=task)


LETTERS = 'ABCDEFGHIJKLMNOP'


def mmlu_pro_rows(data):
    head = (data / 'mp-init.txt').read_text(encoding='utf-8').strip() + '

'
    def body(row):
        return 'Question:
' + row['question'] + '
Options:
' + ''.join(f'{LETTERS[i]}. {o}
' for i, o in enumerate(row['options']))
    shots = {}
    for row in json.loads((data / 'mp-val.json').read_text(encoding='utf-8')):
        text = body(row) + row['cot_content'].replace("A: Let's think step by step.", "Answer: Let's think step by step.") + '

'
        shots.setdefault(row['category'], []).append(text)
    for row in json.loads((data / 'mp-test.json').read_text(encoding='utf-8')):
        cat = row['category']
        yield dict(id=question_id('mmlupro', row['question']), suite='mmlupro', prompt=head.replace('{$}', cat) + ''.join(shots[cat][:5]) + body(row) + "Answer: Let's think step by step.",
                   answer=row['answer'], limit=1024, group=cat)


def super_gpqa_rows(data, sample=2000):
    """A seeded sample by discipline: the whole suite is 26,529 questions, and the sample shows the trend before it is run."""
    text = (data / 'sg-five.yaml').read_text(encoding='utf-8')
    prompt = textwrap.dedent(text[text.index('- |') + 3:].split('
', 1)[1]).strip()
    rows = json.loads((data / 'sg.json').read_text(encoding='utf-8'))
    random.Random(1).shuffle(rows)
    per = {}
    for row in rows:
        per.setdefault(row['discipline'], []).append(row)
    keep = [r for group in per.values() for r in group[:round(sample * len(group) / len(rows))]]
    random.Random(2).shuffle(keep)
    for row in keep:
        question = row['question'] + '
' + ''.join(f'{LETTERS[i]}) {o}
' for i, o in enumerate(row['options']))
        yield dict(id=question_id('supergpqa', row['uuid']), suite='supergpqa', prompt=prompt.replace('{{', '{').replace('}}', '}').replace('{}', question.strip()),
                   answer=row['answer_letter'], limit=1024, group=row['discipline'], calculation=str(row['is_calculation']) == 'True', difficulty=row['difficulty'])


if __name__ == '__main__':
    data, output = Path(sys.argv[1]), Path(sys.argv[2])
    for name, rows in (('std-math.jsonl', math_rows(data)), ('std-bbh.jsonl', bbh_rows(data)), ('std-mmlupro.jsonl', mmlu_pro_rows(data)), ('std-supergpqa.jsonl', super_gpqa_rows(data))):
        rows = list(rows)
        with open(output / name, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        print(name, len(rows), 'empty gold', sum(not r['answer'] for r in rows), re.sub(r'\s+', ' ', rows[0]['prompt'][-120:]))
