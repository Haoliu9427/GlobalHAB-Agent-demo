"""Time-series foundation model comparison. Chronos is not a chat LLM."""
import json,time
import numpy as np

def chronos_scores(X,model_id='amazon/chronos-bolt-small'):
    import torch
    from chronos import BaseChronosPipeline
    torch.set_num_threads(2)
    start=time.perf_counter()
    pipe=BaseChronosPipeline.from_pretrained(model_id,device_map='cpu')
    levels=[.1,.2,.3,.4,.5,.6,.7,.8,.9];allp=[]
    for start_ix in range(0,len(X),64):
        context=torch.as_tensor(X[start_ix:start_ix+64,:,0],dtype=torch.float32)
        quantiles,_=pipe.predict_quantiles(context,prediction_length=1,quantile_levels=levels)
        q=np.asarray(quantiles)[:,0,:]
        # The continuous count forecast is thresholded halfway between 0 and 1 report.
        for v in q: allp.append(float(1-np.interp(np.log1p(.5),np.maximum.accumulate(v),levels,left=.1,right=.9)))
    return np.array(allp),dict(model_id=model_id,parameters=sum(t.numel() for t in pipe.model.parameters()),
        seconds=time.perf_counter()-start,revision=getattr(pipe.model.config,'_commit_hash',None),
        quantile_levels=levels,tail_policy='Probability clipped to trained 0.1–0.9 quantiles, then calibrated on calibration window',
        model_type='pretrained time-series foundation model; not chat LLM',
        pretraining_overlap='Provider pretraining corpus overlap with this public series is not independently excluded; do not call contamination-free.')

def ollama_scores(X,features,model,endpoint='http://localhost:11434/api/generate'):
    """Optional local LLM baseline, strict structured output and no future labels.

    No API key or external upload. Caller must evaluate every common split row;
    any invalid response fails the comparison instead of selecting easy rows.
    """
    import urllib.request
    if not endpoint.startswith(('http://localhost:','http://127.0.0.1:')): raise ValueError('Use a local Ollama endpoint.')
    preds=[];audit=[]
    for i,x in enumerate(X):
        prompt=('Predict probability that next week has at least one reported red-tide event. '
          'Input is past weekly observations, first variable is log1p(report count). '
          'Return JSON only: {"probability": 0.0}. No future outcomes are provided.\n'+
          json.dumps({'features':features,'history':x.tolist()}))
        body=json.dumps({'model':model,'prompt':prompt,'stream':False,'format':'json','options':{'temperature':0,'seed':42}}).encode()
        start=time.perf_counter()
        req=urllib.request.Request(endpoint,data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as r:result=json.load(r)
        value=json.loads(result['response'])['probability']
        if not isinstance(value,(int,float)) or not 0<=value<=1:raise ValueError(f'Invalid LLM probability row {i}')
        preds.append(value);audit.append({'row':i,'seconds':time.perf_counter()-start,'model':model,'response':result['response']})
    return np.array(preds),audit

def hf_llm_scores(X,features,model_id='Qwen/Qwen2.5-0.5B-Instruct',batch_size=1,checkpoint=None):
    """Frozen language-model zero-shot 0/1 logit score, calibrated later.

    No generation/chain-of-thought, prompt search, or future label is used.
    Conditional two-token probability is a score, NOT intrinsically calibrated.
    """
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.set_num_threads(2);torch.manual_seed(42)
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(model_id,padding_side='left')
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,dtype=torch.bfloat16).eval()
    zero=tok.encode('0',add_special_tokens=False);one=tok.encode('1',add_special_tokens=False)
    if len(zero)!=1 or len(one)!=1:raise ValueError('This protocol requires single-token binary labels.')
    template=('Forecast whether next week contains at least one red-tide report. '
        'Rows are past weeks oldest first. log_reports is log(1+report count); '
        'season_sin and season_cos encode the calendar. A zero report is not proof of biological absence. '
        'Answer only 1 (one or more reports) or 0 (no reports).\n')
    import hashlib
    from pathlib import Path
    signature=hashlib.sha256(X.tobytes()+json.dumps([features,model_id,template,
        getattr(model.config,'_commit_hash',None),'round4-bfloat16-token01']).encode()).hexdigest()
    probs=[];tokens=0;previous_seconds=0.
    checkpoint=Path(checkpoint) if checkpoint else None
    if checkpoint and checkpoint.exists():
        cached=json.loads(checkpoint.read_text())
        if cached['signature']!=signature:raise ValueError('LLM checkpoint does not match inputs, model revision or prompt')
        probs=cached['scores'];tokens=cached['tokens'];previous_seconds=cached['seconds']
        if len(probs)>len(X):raise ValueError('Invalid checkpoint length')
    for offset in range(len(probs),len(X),batch_size):
        texts=[]
        for x in X[offset:offset+batch_size]:
            prompt=template+json.dumps({'features':features,'history':[[round(float(v),4) for v in row] for row in x]},separators=(',',':'))
            texts.append(tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True))
        inputs=tok(texts,return_tensors='pt',padding=True);tokens+=int(inputs['attention_mask'].sum())
        with torch.no_grad():
            logits=model(**inputs,logits_to_keep=1).logits[:,-1,[zero[0],one[0]]]
            values=torch.softmax(logits.float(),dim=-1)[:,1].cpu().numpy()
        probs.extend(values.tolist())
        if checkpoint and (len(probs)%8==0 or len(probs)==len(X)):
            temp=checkpoint.with_suffix('.tmp')
            temp.write_text(json.dumps({'signature':signature,'scores':probs,'tokens':tokens,
                'seconds':previous_seconds+time.perf_counter()-start}))
            temp.replace(checkpoint)
        if offset%80==0:print('LLM scored',offset+len(values),'/',len(X),flush=True)
    return np.asarray(probs),{'model_id':model_id,'revision':getattr(model.config,'_commit_hash',None),
        'parameters':sum(p.numel() for p in model.parameters()),'seconds':previous_seconds+time.perf_counter()-start,
        'input_tokens':tokens,'prompt_template':template,'protocol':'frozen zero-shot binary token score; pre-test Platt calibration',
        'pretraining_overlap':'Public data overlap cannot be excluded. This small LLM does not represent all LLMs.'}
