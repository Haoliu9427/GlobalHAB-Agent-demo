"""Run the same user-data workflow as the UI, without changing frozen experiments."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training.user_engine import run
from globalhab_demo.real_training.model_registry import MODELS
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,required=True)
    p.add_argument('--horizon',type=int,choices=[7,14,30],default=7);p.add_argument('--description',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--future',action='store_true');a=p.parse_args()
    if a.output.exists():p.error('Choose a new output file; existing results will not be overwritten.')
    df,forecast,interpretation,manifest,archive=run(a.input.read_bytes(),a.models,a.horizon,a.description,a.epochs,a.future,print)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(archive);print(df.to_string(index=False))
