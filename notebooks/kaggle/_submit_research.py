"""Submit validated notebooks once and record status; never retry automatically."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def main():
    import nbformat
    manifest = json.loads((HERE / 'research_checks/build_manifest.json').read_text())
    for item in manifest:
        path = Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            source = cell.source
            if source.startswith('%%writefile'):
                header, source = source.split('\n', 1)
                name = header.rsplit('/', 1)[-1]
                assert source == (HERE / name).read_text(encoding='utf-8'), name
            compile(source, str(path), 'exec')
    print('ALL THREE NOTEBOOKS: schema, code compilation, embedded-source identity PASS', flush=True)
    cred = json.loads((Path.home() / '.kaggle/kaggle.json').read_text())
    os.environ['KAGGLE_API_TOKEN'] = cred['key']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    record = HERE / 'research_checks/submissions.json'
    records = json.loads(record.read_text()) if record.exists() else []
    for item in manifest[:2]:
        if any(r['kernel'] == item['kernel'] and r.get('notebook_sha256') == item['sha256'] for r in records):
            print('SKIP already attempted:', item['kernel'], flush=True)
            continue
        row = {'kernel': item['kernel'], 'notebook_sha256': item['sha256'], 'status': 'SUBMISSION_ATTEMPTED'}
        records.append(row)
        record.write_text(json.dumps(records, indent=2))
        try:
            result = api.kernels_push(str(Path(item['path']).parent))
            row['response'] = str(result)
            row['status'] = 'SUBMITTED'
            print(item['kernel'], str(result), flush=True)
            status = api.kernels_status(item['kernel'])
            row['remote_status'] = str(status.status)
            row['failure_message'] = str(getattr(status, 'failure_message', ''))
        except Exception as exc:
            row['status'] = 'SUBMISSION_OR_STATUS_ERROR'
            row['error'] = f'{type(exc).__name__}: {exc}'
            print(row, flush=True)
        record.write_text(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
