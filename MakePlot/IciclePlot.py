import plotly.graph_objects as go
import pandas as pd
import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovForest as tbf
from PIL import Image
import numpy as np


wavelet = "db8"
mode = "per"
start_level=3
max_level = 5
beta = 0.49999
y = np.asarray(Image.open("../cloud_dalle.png").convert("L"))/255
y_err = y + np.random.normal(0, 0.1, y.shape)
Image.fromarray(y*255).show()
Image.fromarray(y_err*255).show()

hsm, hde = wt2d.get2DWaveletCoefficients(y_err, wavelet, mode)
tbt = tbf.TwoDimBesovForest(hde, beta, start_level, max_level)
g = tbt.getMinimizingPosteriorCoefficients()

res = []
for el in g:
    if el[0] == 0:
        res.append([f"{el}", abs(g[el]['cH']), f""])
    elif g[el]['cH'] == 0:
        continue
    else:
        res.append([f"{el}", abs(g[el]['cH']), f"({el[0]-1}, {el[1]//4})"])

df = pd.DataFrame(res, columns=["Index", "cH", "Parent"])
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/96c0bd/sunburst-coffee-flavors-complete.csv')
fig = go.Figure(
    go.Icicle(
        labels = df.Index,
        parents = df.Parent,
        root_color="lightgrey",
        values=df.cH,
        marker=dict(
                showscale=True,
                colors=df['cH'],
                colorscale='blues',
                cmid=np.average(df.cH)),
        tiling = dict(
            orientation='v'
        )
    )
)

fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
# fig.show()

inverse = wt2d.inverse2DDWT([hsm, hde], wavelet, mode)
Image.fromarray((inverse*255).astype(np.uint8)).show()