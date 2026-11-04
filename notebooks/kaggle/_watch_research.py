"""Watch submitted full runs; queue the quota-rejected RTDETR once. No training retries."""
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent
CHECKS = HERE / 'research_checks'
OUT = HERE / 'research_results'


def save(path, value):
    """Atomic state save with bounded retries for transient Windows file locks.

    Never fall back to truncating the old state: a failed pre-submission save must
    prevent the submission. Retries here are filesystem-only, never GPU retries.
    """
    import tempfile
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(min(0.25 * 2**attempt, 2))
    finally:
        if temp is not None:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass  # Never mask the original state-save failure.


def main():
    cred = json.loads((Path.home()/'.kaggle/kaggle.json').read_text())
    os.environ['KAGGLE_API_TOKEN'] = cred['key']
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    manifest = json.loads((CHECKS/'build_manifest.json').read_text())
    state_path = CHECKS/'watch_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': {}, 'rtdetr_attempted': False}
    start = time.monotonic()
    while time.monotonic() - start < 18*3600:
        statuses = {}
        for item in manifest:
            ref = item['kernel']
            if ref.endswith('-rtdetr') and not state['rtdetr_attempted']:
                continue
            try:
                status = api.kernels_status(ref)
                text = getattr(status.status, 'name', str(status.status)).upper().split('.')[-1]
                statuses[ref] = text
                row = state['jobs'].setdefault(ref, {})
                row.update(status=text, failure_message=str(getattr(status, 'failure_message', '')))
                print(time.strftime('%Y-%m-%d %H:%M:%S'), ref, text, flush=True)
                if text in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELED') and not row.get('download_attempted'):
                    row['download_attempted'] = True
                    save(state_path, state)
                    dest = OUT / ref.split('/')[-1]
                    dest.mkdir(parents=True, exist_ok=True)
                    try:
                        api.kernels_output(ref, str(dest))
                        row['output'] = str(dest)
                        found = list(dest.rglob('completed.json'))
                        row['completion_records'] = [json.loads(p.read_text()) for p in found]
                        row['artifact_check'] = 'NOT_VERIFIED'
                        if found:
                            root = found[0].parent
                            required = ['preds.json','val_gt.json','results.json','per_class.csv','provenance.json','split.json','config.json','stage_report.json']
                            missing = [name for name in required if not (root/name).is_file()]
                            row['missing_artifacts'] = missing
                            if not missing:
                                result = json.loads((root/'results.json').read_text())
                                split = json.loads((root/'split.json').read_text())
                                row['artifact_check'] = 'PASS' if len(split['val_ids']) == 646 and len(result['per_class_AP50-95']) == 32 else 'FAIL'
                    except Exception:
                        row['download_error'] = traceback.format_exc()
            except Exception:
                state['jobs'].setdefault(ref, {})['status_error'] = traceback.format_exc()
        terminal = {'COMPLETE', 'ERROR', 'CANCELED', 'CANCEL_ACKNOWLEDGED'}
        if not state['rtdetr_attempted'] and any(statuses.get(m['kernel']) in terminal for m in manifest[:2]):
            item = manifest[2]
            path = Path(item['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Do not submit changed unvalidated notebook'
            state['rtdetr_attempted'] = True
            save(state_path, state)  # crash-safe: never duplicate a submission
            try:
                response = api.kernels_push(str(path.parent))
                state['rtdetr_response'] = str(response)
                error = getattr(response, 'error', '')
                if error:
                    state['rtdetr_submission_failed'] = str(error)
                print('RTDETR QUEUE SUBMISSION', response, flush=True)
            except Exception:
                state['rtdetr_submission_failed'] = traceback.format_exc()
        state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        save(state_path, state)
        first_done = all(statuses.get(m['kernel']) in terminal for m in manifest[:2])
        last_done = statuses.get(manifest[2]['kernel']) in terminal or state.get('rtdetr_submission_failed')
        if first_done and last_done:
            print('WATCH COMPLETE; inspect watch_state.json and downloaded artifacts.', flush=True)
            return
        time.sleep(120)
    state['watch_timeout'] = True
    save(state_path, state)
    print('Watcher stopped after 18 hours; no jobs cancelled or retried.', flush=True)

if __name__ == '__main__':
    main()
