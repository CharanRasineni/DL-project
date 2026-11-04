"""Offline watcher regression tests; no Kaggle authentication or submissions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import _watch_research as watcher


class WatcherStateTests(unittest.TestCase):
    def test_transient_windows_lock_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            path.write_text('{"old": true}', encoding='utf-8')
            real_replace = watcher.os.replace
            attempts = []

            def locked_twice(source, destination):
                attempts.append(source)
                if len(attempts) <= 2:
                    raise PermissionError(13, 'Simulated Windows file lock')
                real_replace(source, destination)

            with patch.object(watcher.os, 'replace', side_effect=locked_twice), \
                    patch.object(watcher.time, 'sleep'):
                watcher.save(path, {'new': True})
            self.assertEqual(json.loads(path.read_text()), {'new': True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_permanent_lock_preserves_old_state_and_stops(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            path.write_text('{"rtdetr_attempted": false}', encoding='utf-8')
            with patch.object(watcher.os, 'replace', side_effect=PermissionError('locked')) as replace, \
                    patch.object(watcher.time, 'sleep'):
                with self.assertRaises(PermissionError):
                    watcher.save(path, {'rtdetr_attempted': True})
            self.assertLessEqual(replace.call_count, 8)
            self.assertFalse(json.loads(path.read_text())['rtdetr_attempted'])
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_invalid_json_does_not_replace_previous_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            path.write_text('{"ok": true}', encoding='utf-8')
            with self.assertRaises(ValueError):
                watcher.save(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'ok': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
