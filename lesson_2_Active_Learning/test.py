
from fastai.tabular import *

import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

def entropy(probs):
    """Return the prediction of a T*N*C tensor with :
        - T : the number of samples
        - N : the batch size
        - C : the number of classes
    """
    probs = to_np(probs)
    prob = probs.mean(axis=0)

    entrop = - (np.log(prob) * prob).sum(axis=1)
    return entrop

def get_valid_idx(train_df):
    train_n = len(train_df)

    valid_pct = 0.2
    valid_size = int(valid_pct * train_n)
    valid_idx = np.random.choice(train_n, valid_size, replace=False)
    return valid_idx

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Loading(id="train_df_loading", type="default", children=[
        dcc.Upload(
            id='train_df_upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed',
                'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
            },
        )
    ]),
    dash_table.DataTable(
        id='train_df',
        style_table={
            'maxHeight': '300px',
            'overflowY': 'scroll'
        }),
    html.Button("Train", id="train_button"),


    dcc.Upload(
        id='test_df_upload',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
    ),
    dash_table.DataTable(
        id='test_df',
        style_table={
            'maxHeight': '300px',
            'overflowY': 'scroll'
        }),
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded))


@app.callback([Output('train_df', 'data'),
               Output('train_df', 'columns')],
              [Input('train_df_upload', 'contents')],
              [State('train_df_upload', 'filename')])
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]


@app.callback([Output('test_df', 'data'),
               Output('test_df', 'columns')],
              [Input('test_df_upload', 'contents')],
              [State('test_df_upload', 'filename')])
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]

@app.callback([Output("test_df", 'data')],
              [Input("train_button","n_clicks")],
              [State("test_df","data")])
def train_model(n_clicks,train_df):
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
    learn.fit_one_cycle(5, 1e-1 / 2)

    preds = []
    n_sample = 5

    for i in range(n_sample):
        pred,y = learn.get_preds(ds_type=DatasetType.Test)
        pred = pred[None]
        preds.append(pred)
    preds = torch.cat(preds)

    H = entropy(preds)

    test_df["entropy"] = H
    return test_df

if __name__ == '__main__':
    app.run_server(debug=True)
