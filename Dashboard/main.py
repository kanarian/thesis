# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc,Input, Output, State
import plotly.express as px
import pandas as pd
from PIL import Image
import base64
from copy import deepcopy
import numpy as np

app = Dash(__name__)

noise_added = False

img = Image.open("../cloud_dalle.png")
img_greyscale = img.convert("L")

def b64_image(image_filename):
    with open(image_filename, 'rb') as f:
        image = f.read()
    return 'data:image/png;base64,' + base64.b64encode(image).decode('utf-8')

def image_noise_form():
    return html.Div([
        html.H3("Noisy grescaled image"),
        html.Div(id='noisy-generated-image'),
        dcc.Input(id='noise', type='number', value=0.1, min=0, max=1, step=0.01),
        html.Button(id='submit-val', type="submit", children="Submit", n_clicks=0),
    ])

def wavelet_form():
    if noise_added:
        return html.Div([
        html.Div(["Select a wavelet:", dcc.Dropdown(
            ["Haar","Daubechy 2"],"Haar",id="wavelet-dropdown"),
            html.Div(id="wavelet-dropdown-output")]),
        html.Div(["Select a mode:", dcc.Dropdown(
            ["Periodic","Symmetric","Smooth"],"Periodic",id="mode-dropdown")]),
        html.Div(["Select beta", dcc.Input(id="beta-input", type="number", min=0.0001, max=0.9999, step=0.0001, value=0.5)]),
        ], style={'display': 'flex', "flex-direction": "column", "gap": "20px", "width": "24rem"}),
    else:
        return html.Div("Please generate a noisy image before applying the wavelet transform")

app.layout = html.Div(children=[
    html.H1(children='Upload an image'),
    html.Div([
        html.Div([html.H3('Original image'),html.Img(src=img, width=300, height=300)]),
        html.Div([html.H3('Greyscaled image'), html.Img(src=img_greyscale, width=300, height=300)]),
        image_noise_form(),
        ],
    style={'display': 'flex', "gap": "20px"}),
    wavelet_form(),
    dcc.Store(id="noisy-img-store"),
])

@app.callback(
    Output(component_id="wavelet-dropdown-output", component_property="children"),
    Input(component_id="wavelet-dropdown", component_property="value"),
    suppress_callback_exceptions=True,
)
def update_output(value):
    return f"Selected wavelet: {value}"

@app.callback(
    Output(component_id="noisy-generated-image", component_property="children"),
    Input(component_id="submit-val", component_property="n_clicks"),
    State(component_id="noise", component_property="value")
)
def update_output(n_clicks, noise):
    if n_clicks == 0:
        return "Update the noise value and click submit to see the noisy image"
    noise_added = True
    copy_img = deepcopy(np.asarray(img_greyscale))/255
    noise = np.random.normal(0, noise, copy_img.shape)
    noisy_img_arr = copy_img + noise
    noisy_img = Image.fromarray(noisy_img_arr*255).convert("L")
    return html.Img(src=noisy_img, width=300, height=300)

if __name__ == '__main__':
    app.run_server(debug=True)