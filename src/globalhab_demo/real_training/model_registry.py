"""Explicit executable model inventory. No silent substitution."""
import importlib.util
REMOTE='远程大模型 · API预测'
FOUNDATIONS={
 REMOTE:'server-configured',
 'Chronos-Bolt-tiny':'amazon/chronos-bolt-tiny',
 'Chronos-Bolt-small':'amazon/chronos-bolt-small',
 'Chronos-Bolt-base':'amazon/chronos-bolt-base',
 'Qwen2.5-0.5B-Instruct':'Qwen/Qwen2.5-0.5B-Instruct',
 'Qwen2.5-1.5B-Instruct':'Qwen/Qwen2.5-1.5B-Instruct',
 'Qwen2.5-3B-Instruct':'Qwen/Qwen2.5-3B-Instruct',
 'SmolLM2-360M-Instruct':'HuggingFaceTB/SmolLM2-360M-Instruct',
}
TABULAR=['Logistic','GAM (Spline Logistic)','Gaussian Naive Bayes','kNN','RBF-SVM','Decision Tree','Random Forest','Extra Trees','AdaBoost','Gradient Boosting','HistGradientBoosting','MLP','XGBoost','LightGBM']
BASELINES=['Seasonal Climatology','Event Persistence']
DEEP=['Lightweight TCN','EcoTemporalNet','STS-Interaction GLM','STS-Gated TCN']
FUSIONS={'EcoFusion':['EcoTemporalNet','HistGradientBoosting']}
FUSIONS.update({n+' + EcoFusion':[n,'EcoFusion'] for n in FOUNDATIONS})
MODELS=BASELINES+TABULAR+DEEP+list(FOUNDATIONS)+list(FUSIONS)
SCIENCE_COLUMNS=['local_raw_signal','upstream_raw_signal','circulation_residence_proxy','nitrate','phosphate','silicate']
OTHER={'分子物种检出率基线':'需要分子调查的物种与检测方法字段；请使用中国近海调查模块。',
 '空间Durbin模型':'空间效应解释，需空间权重矩阵；不是统一二分类预测器。',
 'TE/CTE信息流':'方向及时滞检验；不输出同一目标的分类概率。',
 '生物响应过程模型':'生理压力情景模拟；请进入生物响应沙盘。',
 '养殖风险投影':'危害、暴露与脆弱性组合；请进入风险研判。',
 'Bayesian EI/IG、Thompson、Random探索策略':'实验选择策略，不是观测分类器；请进入探索与验证。'}

def expand(selected):
    out=[]
    def add(n):
        if n not in MODELS:raise ValueError('不支持的模型：'+n)
        for child in FUSIONS.get(n,[]):add(child)
        if n not in out:out.append(n)
    for n in selected:add(n)
    return out

def missing(name):
    modules=[]
    for n in expand([name]):
        if n in ['EcoTemporalNet','Lightweight TCN'] or (n in FOUNDATIONS and n!=REMOTE):modules+=['torch']
        if n in FOUNDATIONS and n!=REMOTE:modules+=['chronos' if n.startswith('Chronos') else 'transformers']
        if n=='XGBoost':modules+=['xgboost']
        if n=='LightGBM':modules+=['lightgbm']
    return sorted({m for m in modules if importlib.util.find_spec(m) is None})

def estimator(name,seed,ntrain):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer,StandardScaler
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    if name=='Logistic':return LogisticRegression(C=.2,max_iter=1000,random_state=seed)
    if name=='GAM (Spline Logistic)':return make_pipeline(SplineTransformer(n_knots=4,include_bias=False),StandardScaler(),LogisticRegression(C=.3,max_iter=1000))
    if name=='Gaussian Naive Bayes':return GaussianNB()
    if name=='kNN':return KNeighborsClassifier(n_neighbors=min(15,ntrain),weights='distance')
    if name=='RBF-SVM':return SVC(C=1,probability=True,random_state=seed)
    if name=='Decision Tree':return DecisionTreeClassifier(max_depth=5,min_samples_leaf=4,random_state=seed)
    if name=='Random Forest':return RandomForestClassifier(n_estimators=150,max_depth=12,min_samples_leaf=10,n_jobs=2,random_state=seed)
    if name=='Extra Trees':return ExtraTreesClassifier(n_estimators=150,min_samples_leaf=4,n_jobs=2,random_state=seed)
    if name=='AdaBoost':return AdaBoostClassifier(n_estimators=80,learning_rate=.05,random_state=seed)
    if name=='Gradient Boosting':return GradientBoostingClassifier(n_estimators=100,max_depth=2,random_state=seed)
    if name=='HistGradientBoosting':return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=5,early_stopping=False,random_state=seed)
    if name=='MLP':return MLPClassifier(hidden_layer_sizes=(24,),max_iter=250,random_state=seed)
    if name=='XGBoost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=160,max_depth=3,n_jobs=2,random_state=seed,eval_metric='logloss')
    if name=='LightGBM':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=160,num_leaves=15,n_jobs=2,verbosity=-1,random_state=seed)
    raise ValueError(name)
