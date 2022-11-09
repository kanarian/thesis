import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovForest as tbf
from dash import Dash, html, dcc,Input, Output, State
import plotly.express as px
import pandas as pd
from PIL import Image
import base64
from copy import deepcopy
import numpy as np
import json

app = Dash(__name__)

img = Image.open("../cloud_dalle.png")
img_greyscale = img.convert("L")
def image_noise_form():
    return html.Div([
        html.H3("Noisy grescaled image"),
        html.Div(id='noisy-generated-image'),
        dcc.Input(id='noise', type='number', value=0.1, min=0, max=1, step=0.01),
        html.Button(id='submit-val', type="submit", children="Submit", n_clicks=0),
    ])

def wavelet_form():
    return html.Div([
    html.Div(["Select a wavelet:", dcc.Dropdown(
        ["Haar","Daubechy 2"],"Haar",id="wavelet-dropdown")]),
    html.Div(["Select a mode:", dcc.Dropdown(
        ["Periodic","Symmetric","Smooth"],"Periodic",id="mode-dropdown")]),
    html.Div(["Select beta", dcc.Input(id="beta-input", type="number", min=0.0001, max=0.9999, step=0.0001, value=0.5)]),
    html.Div(["Select start level", dcc.Input(id="start-level-input", type="number", min=0, max=10, step=1, value=3)]),
    html.Div(["Select max level", dcc.Input(id="max-level-input", type="number", min=0, max=10, step=1, value=5)]),
    html.Button(id='submit-wavelet', type="submit", children="Submit", n_clicks=0),
    ], id="wavelet-form")

app.layout = html.Div(children=[
    html.H1(children='Upload an image'),
    html.Div([
        html.Div([html.H3('Original image'),html.Img(src=img, width=300, height=300)]),
        html.Div([html.H3('Greyscaled image'), html.Img(src=img_greyscale, width=300, height=300)]),
        image_noise_form(),
        ],
    style={'display': 'flex', "gap": "20px"}),
    wavelet_form(),
    html.Div(id='wavelet-transform-output'),
    dcc.Store(id="noisy-img-store"),
])


@app.callback(
    Output(component_id="noisy-img-store", component_property="data"),
    Input(component_id="submit-val", component_property="n_clicks"),
    State(component_id="noise", component_property="value")
)
def update_noisy_img_store(n_clicks, noise):
    if n_clicks == 0:
        return "Update the noise value and click submit to see the noisy image"
    copy_img = deepcopy(np.asarray(img_greyscale))/255
    noise = np.random.normal(0, noise, copy_img.shape)
    noisy_img_arr = (copy_img + noise)*255
    to_show = {"noisy_img": noisy_img_arr.tolist(), "noise_added": True}
    return json.dumps(to_show)

@app.callback(
    Output(component_id="noisy-generated-image", component_property="children"),
    Input(component_id="noisy-img-store", component_property="data"))
def update_noisy_generated_image(data):
    if "noisy_img" not in data:
        return "Update the noise value and click submit to see the noisy image"
    json_data = json.loads(data)
    new_img = Image.fromarray(np.array(json_data["noisy_img"])).convert("L")
    return html.Img(src=new_img, width=300, height=300)

@app.callback(
    Output(component_id="wavelet-form", component_property="style"),
    Input(component_id="noisy-img-store", component_property="data"))
def update_wavelet_form(data):
    if "noisy_img" not in data:
        return {"display": "none"}
    return {"display": "flex", "flex-direction": "column", "gap": "20px", "width": "24rem"}

@app.callback(
    Output(component_id="wavelet-transform-output", component_property="children"),
    Input(component_id="submit-wavelet", component_property="n_clicks"),
    [State(component_id="wavelet-dropdown", component_property="value"),
     State(component_id="mode-dropdown", component_property="value"),
     State(component_id="beta-input", component_property="value"),
     State(component_id="start-level-input", component_property="value"),
     State(component_id="max-level-input", component_property="value"),
     State(component_id="noisy-img-store", component_property="data")],
    prevent_initial_call=True)
def update_wavelet_dropdown_output(n_clicks, wavelet, mode, beta, start_level, max_level, data):
    print("wavelet transform start")
    wavelet_map = {"Haar": "haar", "Daubechy 2": "db2"}
    mode_map = {"Periodic": "per", "Symmetric": "symm", "Smooth": "smooth"}
    y_err = np.array(json.loads(data)["noisy_img"])/255
    hsm, hde = wt2d.get2DWaveletCoefficients(y_err, wavelet_map[wavelet], mode_map[mode])
    tbt = tbf.TwoDimBesovForest(hde, beta, start_level, max_level)
    g = tbt.getMinimizingPosteriorCoefficients()
    y = wt2d.inverse2DDWT([hsm, g], wavelet_map[wavelet], mode_map[mode])
    recon_img = Image.fromarray(y*255).convert("L")
    print("wavelet transform finish")
    return html.Img(src=recon_img, width=300, height=300)

if __name__ == '__main__':
    app.run_server(debug=True)