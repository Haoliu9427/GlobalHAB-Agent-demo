"""python scripts/train_real.py --dataset hong_kong --foundation"""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from globalhab_demo.real_training import data
from globalhab_demo.real_training.benchmark import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['habsos','hong_kong','china_field'],required=True)
    p.add_argument('--input',type=Path);p.add_argument('--horizon',type=int,default=7,choices=[7,14,30]);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seeds',default='17,42,73');p.add_argument('--foundation',action='store_true');p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--output',type=Path);p.add_argument('--cutoffs',nargs=3);a=p.parse_args()
    raw=ROOT/'data/training_real/raw'
    if a.dataset=='habsos':ds=data.habsos(a.input or raw/'habsos.csv.gz',a.horizon)
    elif a.dataset=='hong_kong':
        if a.horizon!=7:p.error('HK endpoint is next-week report occurrence; only --horizon 7 is defined.')
        ds=data.hong_kong(a.input or raw/'hong_kong_red_tide.csv')
    else:
        if a.input is None:p.error('china_field requires --input with verified observations')
        ds=data.field(a.input,a.horizon)
    dates=a.cutoffs or (['2013-01-01','2017-01-01','2020-01-01'] if a.dataset=='hong_kong' else None)
    output=a.output or ROOT/'outputs/real_training'/ds.task
    if (output/'metrics.csv').exists():p.error('Results already exist; choose a new --output to preserve frozen evaluation.')
    run(ds,output,dates,tuple(map(int,a.seeds.split(','))),a.epochs,a.foundation,a.bootstrap)
if __name__=='__main__':main()
