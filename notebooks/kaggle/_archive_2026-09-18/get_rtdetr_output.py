
import json, requests, os

flake = json.load(open(r'C:\Users\rshit\.kaggle\kaggle.json'))
key = flake['key']
headers = {'Authorization': f'Bearer {key}'}
API = 'https://www.kaggle.com/api/v1/kernels'

r = requests.get(f'{API}/eternalsorcerer/fathomnet-05-rtdetr/view', headers=headers)
print('view status:', r.status_code)
if r.status_code != 200:
    print(r.text[:400])
    raise SystemExit(1)
v = r.json()
k_id = v.get('id')
print('kernel id:', k_id)
runs = v.get('kernelRuns', [])
print('num runs:', len(runs))
for run in runs[-2:]:
    print('  run', run.get('id'), run.get('status'), run.get('dateRunStarted'))

run_id = None
for run in reversed(runs):
    if run.get('status') == 'complete':
        run_id = run['id']
        break
if not run_id:
    run_id = runs[-1]['id'] if runs else None
print('using run_id:', run_id)
if not run_id:
    raise SystemExit(1)

r2 = requests.get(f'{API}/eternalsorcerer/fathomnet-05-rtdetr/view/output/{run_id}', headers=headers)
print('output view status:', r2.status_code)
if r2.status_code == 200:
    out = r2.json()
    files = out.get('files', out if isinstance(out, list) else [])
    print('num files:', len(files))
    outdir = r'C:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle\05_rtdetr\output'
    os.makedirs(outdir, exist_ok=True)
    for f in files:
        name = f.get('name') or f.get('fileName')
        size = f.get('sizeBytes', 0)
        if not name:
            continue
        path = os.path.join(outdir, name)
        print(f'  {name}  ({size} bytes)', flush=True)
        if name == 'log':
            rr = requests.get(f'{API}/eternalsorcerer/fathomnet-05-rtdetr/download/{run_id}/{name}', headers=headers)
            print('  log download status:', rr.status_code, flush=True)
            if rr.status_code == 200:
                with open(path, 'wb') as fh:
                    fh.write(rr.content)
                print(f'    -> saved ({len(rr.content)} bytes)')
            else:
                print(f'    -> download failed {rr.status_code}: {rr.text[:150]}')
        else:
            print(f'    -> (binary, size {size})')
else:
    print(r2.text[:400])
