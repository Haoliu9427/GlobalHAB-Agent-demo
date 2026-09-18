import json
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from globalhab_demo.real_training import remote_qwen as q
from globalhab_demo.real_training.model_registry import REMOTE,missing
C={'base_url':'https://example.org/v1','api_key':'secret-test','model':'qwen-test'}
def response(content,status=200):
    return SimpleNamespace(status_code=status,json=lambda:{'choices':[{'message':{'content':content}}],'usage':{'total_tokens':10}})
def test_no_torch_dependency():assert missing(REMOTE)==[]
def test_payload():
    ds=SimpleNamespace(X=np.array([[[1.,2.]]]),features=['a','b'])
    with patch.object(q.requests,'post',return_value=response('{"probability":0.3}')) as post:
        p,m=q.score(C,ds,[0],7,'threshold',lambda x:None)
        assert p[0]==.3 and m['reported_total_tokens']==10
        sent=post.call_args.kwargs
        assert sent['allow_redirects'] is False
        data=json.loads(sent['json']['messages'][1]['content'])
        assert set(data)=={'event_definition','horizon_days','target','features','past_history'}
        assert 'secret-test' not in json.dumps(m)
@pytest.mark.parametrize('text',['{"probability":2}','{"probability":true}','nonsense','{"probability":NaN}'])
def test_invalid(text):
    ds=SimpleNamespace(X=np.zeros((1,1,1)),features=['a'])
    with patch.object(q.requests,'post',return_value=response(text)):
        with pytest.raises(ValueError):q.score(C,ds,[0],7,'x',lambda x:None)
def test_error():
    with patch.object(q.requests,'post',return_value=response('secret-test',401)):
        with pytest.raises(ValueError) as e:q.chat(C,[])
        assert 'secret-test' not in str(e.value)
def test_https():
    with pytest.raises(ValueError):q.chat(dict(C,base_url='http://example.org'),[])
def test_remote_engine_common_split():
    import io,runpy,zipfile
    from pathlib import Path
    import pandas as pd
    from globalhab_demo.real_training.user_engine import run
    fixture=runpy.run_path(str(Path(__file__).with_name('test_own_observations.py')))['fixture']
    data=pd.read_csv(io.BytesIO(fixture())).head(700).to_csv(index=False).encode()
    with patch.object(q.requests,'post',return_value=response('{"probability":0.3}')):
        table,forecast,_,_,archive=run(data,[REMOTE,'Logistic'],description='Synthetic test only',future=True,remote_config=C)
    assert REMOTE in set(table.model)
    assert forecast.y.isna().all()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        predictions=pd.read_csv(z.open('predictions.csv.gz'),compression='gzip')
        sets=predictions.groupby(['model','seed']).sample_id.apply(set)
        assert all(s==sets.iloc[0] for s in sets)
        assert all(b'secret-test' not in z.read(n) for n in z.namelist())

@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(q.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('93.184.216.34',443))])

def test_blocks_private_endpoint(monkeypatch):
    monkeypatch.setattr(q.socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('127.0.0.1',443))])
    with pytest.raises(ValueError):q.validate_endpoint('https://localhost/v1')

def test_user_config_ui():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from globalhab_demo.real_training.user_workbench import render\nrender()',default_timeout=180).run()
    assert not app.exception
    app.text_input(key='user_api_url').set_value('https://example.org/v1')
    app.text_input(key='user_api_model').set_value('other-compatible-model')
    app.text_input(key='user_api_key').set_value('private-key').run()
    assert not app.exception
    assert app.text_input(key='user_api_key').proto.type==1
    other=AppTest.from_string('from globalhab_demo.real_training.user_workbench import render\nrender()',default_timeout=180).run()
    assert other.text_input(key='user_api_key').value==''
    next(b for b in app.button if b.label=='清除凭证').click().run()
    assert app.text_input(key='user_api_key').value==''

def test_deepseek_provider():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from globalhab_demo.real_training.user_workbench import render\nrender()',default_timeout=180).run()
    app.text_input(key='user_api_key').set_value('old-key').run()
    app.selectbox(key='remote_provider').select('DeepSeek').run()
    assert not app.exception
    assert app.text_input(key='user_api_url').value=='https://api.deepseek.com'
    assert app.text_input(key='user_api_key').value==''

def test_list_models():
    reply=SimpleNamespace(status_code=200,json=lambda:{'data':[{'id':'account-model'}]})
    with patch.object(q.requests,'get',return_value=reply) as get:
        assert q.list_models(C)==['account-model']
        assert get.call_args.args[0]=='https://example.org/v1/models'
        assert get.call_args.kwargs['allow_redirects'] is False

def test_deepseek_stable_disables_default_thinking():
    deep={'base_url':'https://api.deepseek.com','api_key':'secret-test','model':'deepseek-flash'}
    with patch.object(q.requests,'post',return_value=response('可见最终回答')) as post:
        text,_=q.chat(deep,[{'role':'user','content':'解释结果'}],max_tokens=500,thinking_mode='stable')
        assert text=='可见最终回答'
        payload=post.call_args.kwargs['json']
        assert payload['thinking']=={'type':'disabled'}
        assert payload['reasoning_effort']=='none'
        assert payload['stream'] is False


def test_deepseek_thinking_empty_content_retries_stable():
    deep={'base_url':'https://api.deepseek.com','api_key':'secret-test','model':'deepseek-flash'}
    first=SimpleNamespace(status_code=200,json=lambda:{
        'choices':[{'message':{'content':None,'reasoning_content':'internal reasoning'},'finish_reason':'length'}],
        'usage':{'total_tokens':5000},
    })
    second=response('重试后的最终回答')
    with patch.object(q.requests,'post',side_effect=[first,second]) as post:
        text,_=q.chat(deep,[{'role':'user','content':'解释结果'}],max_tokens=500,thinking_mode='high')
        assert text=='重试后的最终回答'
        assert post.call_count==2
        retry_payload=post.call_args_list[1].kwargs['json']
        assert retry_payload['thinking']=={'type':'disabled'}
        assert retry_payload['reasoning_effort']=='none'
        assert retry_payload['max_tokens']>=2200


def test_chat_accepts_text_block_content():
    block=SimpleNamespace(status_code=200,json=lambda:{
        'choices':[{'message':{'content':[{'type':'text','text':'第一段'},{'type':'text','text':'第二段'}]},'finish_reason':'stop'}],
        'usage':{},
    })
    with patch.object(q.requests,'post',return_value=block):
        text,_=q.chat(C,[{'role':'user','content':'x'}])
    assert text=='第一段\n第二段'
