import codecs
import json
import re
from pathlib import Path
log = Path(r'c:\Users\rshit\OneDrive\Documents\Charan DL project\notebooks\kaggle\research_results\fasterrcnn_failed\inline.log')
text = log.read_text(encoding='utf-8')
parts = re.findall(r'"data":"((?:[^"\\]|\\.)*)"', text)
decoded = []
for part in parts:
    try:
        decoded.append(json.loads('"' + part + '"'))
    except Exception:
        decoded.append(codecs.decode(part, 'unicode_escape'))
stream = ''.join(decoded)
out = log.parent / 'decoded.log'
out.write_text(stream, encoding='utf-8', errors='replace')
print('parts:', len(parts), 'chars:', len(stream))
print(stream[-7000:])
