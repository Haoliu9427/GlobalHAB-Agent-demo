"""Run the same user-data workflow as the UI, without changing frozen experiments."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training.own_observations import execute,CLASSICAL,FOUNDATION
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=CLASSICAL+FOUNDATION,required=True)
    p.add_argument('--horizon',type=int,choices=[7,14,30],default=7);p.add_argument('--description',required=True);p.add_argument('--epochs',type=int,default=20)
    a=p.parse_args()
    if a.output.exists():p.error('Choose a new output file; existing results will not be overwritten.')
    df,manifest,archive=execute(a.input.read_bytes(),a.models,a.horizon,a.description,a.epochs,print)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(archive);print(df.to_string(index=False))
