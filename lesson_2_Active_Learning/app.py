import streamlit as st
from fastai.tabular import *
from fastai_bayesian.custom_dropout import CustomDropout, switch_custom_dropout, convert_layers
from fastai_bayesian.metrics import (entropy, BALD, uncertainty_best_probability,
                      get_preds_sample, plot_hist_groups, top_k_uncertainty)
import SessionState  # Assuming SessionState.py lives on this folder

from fastai.callback import Callback
import streamlit as st

from st_helper import ProgressCallback, plot_confusion_matrix

session = SessionState.get(run_id=0)

@st.cache
def load_data(fn):
    data = pd.read_csv(fn)
    return data

def get_valid_idx(train_df):
    train_n = len(train_df)

    valid_pct = 0.2
    valid_size = int(valid_pct * train_n)
    valid_idx = np.random.choice(train_n, valid_size, replace=False)
    return valid_idx

st.title("Fastai Active learner")

st.header('Training data')
train_df = load_data("train_df.csv")
test_df = load_data("test_df.csv")
st.dataframe(train_df)

st.header("Model training")

if st.button("Start training"):
    # Update the session
    session.run_id += 1

    valid_idx = get_valid_idx(train_df)

    procs = [FillMissing, Categorify, Normalize]
    dep_var = 'salary'
    cat_names = ['workclass', 'education', 'marital-status',
                 'occupation', 'relationship', 'race', 'sex', 'native-country']

    path = Path()
    data = TabularDataBunch.from_df(path, train_df,
        dep_var, valid_idx=valid_idx, procs=procs,cat_names=cat_names,
        test_df=test_df)
        
    learn = tabular_learner(data, layers=[200,100],ps=[0.10,0.05], emb_szs={'native-country': 10}, metrics=accuracy)
    
    learn.fit_one_cycle(5, 1e-1 / 2,callbacks=progress)

    pred,y = learn.get_preds()
    classes = learn.data.train_ds.y.classes
    _, c = np.unique(classes, return_inverse=True)

    fig = plot_confusion_matrix(y,pred.argmax(dim=1),c,figsize=(8,8),normalize=True)

    st.pyplot(fig)
