"""Verify the delivered source/data/model bytes without any network access."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'PACKAGE_SHA256.json'
if not manifest.exists():raise SystemExit('PACKAGE_SHA256.json is missing')
entries=json.loads(manifest.read_text());bad=[]
for name,digest in entries.items():
    p=ROOT/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:bad.append(name)
print(json.dumps({'checked_files':len(entries),'missing_or_changed':bad},ensure_ascii=False,indent=2))
sys.exit(1 if bad else 0)
