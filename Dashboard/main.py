import dash

import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovForest as tbf
from dash import Dash, html, dcc,Input, Output, State
from PIL import Image
from copy import deepcopy
import numpy as np
import json
import pywt
import base64
from io import BytesIO

app = Dash(__name__)

z = pywt.families()
possible_waves = []
wavelist = pywt.wavelist(kind="discrete")
for el in wavelist:
    thisWavelet = pywt.Wavelet(el)
    possible_waves.append({"code": el, "name":f"{thisWavelet.family_name}-{thisWavelet.dec_len}"})

def image_decode(content):
    content_type, content_string = content.split(',')
    # im_b64 = base64.b64encode(content_string)
    im_bytes = base64.b64decode(content_string)
    im_file = BytesIO(im_bytes)
    return im_file

def load_and_preprocess(content):
    image = image_decode(content)
    image1 = Image.open(image)
    rgb = Image.new('RGB', image1.size)
    rgb.paste(image1)
    image = rgb
    test_image = image.resize((512,512))
    return test_image


def image_noise_form():
    return html.Div([
        html.H3("Noisy grescaled image"),
        html.Div(id='noisy-generated-image'),
        dcc.Input(id='noise', type='number', value=0.1, min=0, max=1, step=0.01),
        html.Button(id='submit-val', type="submit", children="Submit", n_clicks=0),
    ],id="image-noise-form")

def wavelet_form():
    return html.Div([
    html.Div(["Select a wavelet:", dcc.Dropdown(
        [val['name'] for val in possible_waves],"Haar-2",id="wavelet-dropdown")]),
    html.Div(["Select a mode:", dcc.Dropdown(
        ["Periodic","Symmetric","Smooth"],"Periodic",id="mode-dropdown")]),
    html.Div(["Select beta", dcc.Input(id="beta-input", type="number", min=0.0001, max=0.9999, step=0.0001, value=0.5)]),
    html.Div(["Select start level", dcc.Input(id="start-level-input", type="number", min=0, max=10, step=1, value=3)]),
    html.Div(["Select max level", dcc.Input(id="max-level-input", type="number", min=0, max=10, step=1, value=5)]),
    html.Button(id='submit-wavelet', type="submit", children="Submit", n_clicks=0),
    ], id="wavelet-form")

app.layout = html.Div(children=[
    html.H1(children='Two-Dimensional Besov Tree Wavelet Denoising'),
    html.Div([html.H3("Upload your own image or select a test image"),
             dcc.Dropdown(options={"../cloud_dalle.png" : "Dall-E generated Cloud", "../Koala.jpg" : "Koala"},value="", id="test-image-dropdown"),
             dcc.Upload(id='upload-image',
                        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                     style={'cursor': 'pointer','width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'}),
             ]),
    html.Div([
        html.Div(id="original-image-div"),
        html.Div(id="greyscale-image-div"),
        image_noise_form(),
        ],
    style={'display': 'flex', "gap": "20px"}),
    wavelet_form(),
    dcc.Store(id="image-store"),
    dcc.Store(id="noisy-img-store"),
    dcc.Store(id="wavelet-transform-store"),
    html.Div(id='wavelet-transform-container'),
])

@app.callback(
    Output("image-noise-form", "style"),
    Input("image-store", "data")
)
def show_image_noise_form(data):
    if data is None:
        return {"display": "none"}
    else:
        return {"display": "block"}

@app.callback(
    Output("image-store", "data"),
    [Input("test-image-dropdown", "value"),
     Input("upload-image", "contents")])
def load_test_image(image_path, contents):
    if dash.callback_context.triggered_id == "test-image-dropdown":
        if image_path == "":
            return None
        img = Image.open(image_path).convert("RGB")
        imgStore = {"image": np.array(img).tolist()}
        return json.dumps(imgStore)
    elif dash.callback_context.triggered_id == "upload-image":
        if contents is not None:
            image = load_and_preprocess(contents)
            imgStoreObj = {"image": np.array(image).tolist()}
            return json.dumps(imgStoreObj)


@app.callback(
    Output("original-image-div", "children"),
    Input("image-store", "data"))
def display_original_image(image_store):
    if image_store is None:
        return None
    imgStore = json.loads(image_store)
    img = Image.fromarray(np.array(imgStore["image"]).astype(np.uint8))
    return html.Div([html.H3("Original img"), html.Img(src=img,height=300,width=300)])

@app.callback(
    Output("greyscale-image-div", "children"),
    Input("image-store", "data"))
def display_original_image(image_store):
    if image_store is None:
        return None
    imgStore = json.loads(image_store)
    img = Image.fromarray(np.array(imgStore["image"]).astype(np.uint8)).convert("L")
    return html.Div([html.H3("Greyscale img"), html.Img(src=img,height=300,width=300)])


@app.callback(
    Output("mode-dropdown", "options"),
    Input("wavelet-dropdown", "value"))
def update_mode_dropdown(wavelet):
    if wavelet == "Haar-2":
        return ["Periodic", "Symmetric", "Smooth"]
    else:
        return ["Periodic"]


@app.callback(
    Output(component_id="noisy-img-store", component_property="data"),
    Input(component_id="submit-val", component_property="n_clicks"),
    [State(component_id="noise", component_property="value"),
     State(component_id="image-store", component_property="data")]
)
def update_noisy_img_store(n_clicks, noise, imgStoreData):
    if imgStoreData is None:
        return "Please select or upload an image"
    if n_clicks == 0:
        return "Update the noise value and click submit to see the noisy image"
    imgStore = json.loads(imgStoreData)
    img_arr = Image.fromarray(np.array(imgStore["image"]).astype(np.uint8)).convert("L")
    copy_img = deepcopy(np.asarray(img_arr))/255
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
    Output(component_id="wavelet-transform-store", component_property="data"),
    Input(component_id="submit-wavelet", component_property="n_clicks"),
    [State(component_id="wavelet-dropdown", component_property="value"),
     State(component_id="mode-dropdown", component_property="value"),
     State(component_id="beta-input", component_property="value"),
     State(component_id="start-level-input", component_property="value"),
     State(component_id="max-level-input", component_property="value"),
     State(component_id="noisy-img-store", component_property="data"),
     State(component_id="wavelet-transform-store", component_property="data")],
    prevent_initial_call=True)
def update_wavelet_dropdown_output(n_clicks, wavelet, mode, beta, start_level, max_level, dataNoisyImg, dataWaveletTransformStore):
    print("wavelet transform start")
    wavelet = list(filter(lambda x: x["name"] == wavelet, possible_waves))[0]["code"]
    mode_map = {"Periodic": "per", "Symmetric": "symm", "Smooth": "smooth"}
    y_err = np.array(json.loads(dataNoisyImg)["noisy_img"])/255
    hsm, hde = wt2d.get2DWaveletCoefficients(y_err, wavelet, mode_map[mode])
    tbt = tbf.TwoDimBesovForest(hde, beta, start_level, max_level)
    g = tbt.getMinimizingPosteriorCoefficients()
    y = wt2d.inverse2DDWT([hsm, g], wavelet, mode_map[mode])
    print("wavelet transform finish")
    this_wt_data = {"recon_img" : y.tolist(), "wavelet_transform_params": {"wavelet": wavelet, "mode": mode, "beta": beta, "start_level": start_level, "max_level": max_level}}

    if dataWaveletTransformStore is None:
        return json.dumps([this_wt_data])

    wt_data = json.loads(dataWaveletTransformStore)

    new_wt_data = wt_data
    new_wt_data.append(this_wt_data)

    return json.dumps(new_wt_data)

@app.callback(
Output(component_id="wavelet-transform-container", component_property="children"),
Input(component_id="wavelet-transform-store", component_property="data"))
def update_wavelet_transform_container(data):
    if data is None:
        return "Update the wavelet transform parameters and click submit to see the reconstructed image"
    json_data = json.loads(data)
    new_divs = []
    for el in reversed(json_data):
        if "recon_img" in el:
            [wavelet, mode, beta, start_level, max_level] = [el["wavelet_transform_params"][key] for key in ["wavelet", "mode", "beta", "start_level", "max_level"]]
            recon_img = Image.fromarray(np.array(el["recon_img"])*255).convert("L")
            new_divs.append(html.Div([html.H3(f"w:{wavelet}-m:{mode}-b:{beta}-s:{start_level}-m:{max_level}"),html.Img(src=recon_img, width=300, height=300)]))
    return new_divs

    # if children == None:
    #     return html.Div([html.H3("Reconstructed image"), html.Img(src=recon_img, width=300, height=300)])
    # child_to_add = html.Div([html.H3(f"Reconstructed image {wavelet}-{mode}-{beta}-{start_level}-{max_level}"),
    #                          html.Img(src=recon_img, width=300, height=300)])
    # child_to_add_json = child_to_add.to_plotly_json()
    # return

if __name__ == '__main__':
    app.run_server(debug=True)