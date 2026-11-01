db = {'alice': {'age': 30, 'tags': ['a', 'b']}, 'bob': {'age': 25, 'tags': []}}
db['carol'] = {'age': 35, 'tags': ['c']}
for name, rec in sorted(db.items(), key=lambda kv: kv[1]['age']):
    print(name, rec['age'], ','.join(rec['tags']) or '-', len(rec['tags']))
print({k: v['age'] for k, v in db.items() if v['age'] > 26}, sum(r['age'] for r in db.values()) / len(db), [k for k, v in db.items() if 'a' in v['tags']], max(db, key=lambda k: db[k]['age']), db.get('dave', {}).get('age', 'n/a'))
db['alice']['tags'].append('z')
copy = {k: dict(v) for k, v in db.items()}
copy['alice']['tags'].append('shared')
copy['bob']['age'] = 99
print(db['alice']['tags'], db['bob']['age'])
