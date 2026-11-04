import csv, json
from pathlib import Path

k = Path('notebooks/kaggle')


def dump_nb(rel):
    nb = json.load(open(k / rel / rel.split('/')[-1] + '.ipynb'))
    print('\n' + '=' * 72)
    print(f'### {rel}  ({len(nb["cells"])} cells) ###')
    for i, c in enumerate(nb['cells']):
        kind = c.get('cell_type', '?')
        src = ''.join(c.get('source', []))
        label = 'markdown' if kind == 'markdown' else f'code ({len(src)} chars)'
        print(f'\n===== CELL {i} [{label}] =====')
        print(src)


dump_nb('02_split')
dump_nb('03_fasterrcnn')
dump_nb('04_yolov8s')
dump_nb('05_rtdetr')

print('\n' + '=' * 40)
print('=== class-counts CSV ===')
rows = list(csv.reader(open(k / '02_split/output/table_split_class_counts.csv')))
print('header:', rows[0])
for r in rows[1:]:
    print(', '.join(r))
tot_t = sum(int(r[1]) for r in rows[1:])
tot_v = sum(int(r[2]) for r in rows[1:])
print(f'train={tot_t} val={tot_v} total={tot_t + tot_v}')
