"""The prompts of the published few-shot protocols, as rows for external_bench --rows. MATH: the four Minerva exemplars of Lewkowycz et al. 2022.
BBH: the three CoT exemplars of Suzgun et al. 2022. standard_rows.py <data dir> <output dir>: the data dir holds what was fetched, read as data."""
import ast, json, re, sys
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


if __name__ == '__main__':
    data, output = Path(sys.argv[1]), Path(sys.argv[2])
    for name, rows in (('std-math.jsonl', math_rows(data)), ('std-bbh.jsonl', bbh_rows(data))):
        rows = list(rows)
        with open(output / name, 'w', encoding='utf-8', newline='\n') as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        print(name, len(rows), 'empty gold', sum(not r['answer'] for r in rows), re.sub(r'\s+', ' ', rows[0]['prompt'][-120:]))
