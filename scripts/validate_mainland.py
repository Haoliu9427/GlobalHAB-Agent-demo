"""Reproduce mainland coastal survey validation from checksum-verified original files."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training.mainland import run
if __name__=='__main__':
    for marker in ['ITS1','18S V4']:run(ROOT,marker)
