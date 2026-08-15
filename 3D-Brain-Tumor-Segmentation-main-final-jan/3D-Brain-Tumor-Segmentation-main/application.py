import random
import os
import pandas as pd
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import tensorflow as tf
import plotly.graph_objs as go
import nibabel as nib
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import config
from Model import DiceCoefficientLoss
import plotly
from quantification import compute_tissue_quantification, overlay_segmentation, bar_chart_from_metrics, pie_chart_from_metrics, quantification_dataframe, save_quantification_csv

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import datetime
import json
import io
import base64

"""#### Loading model"""


def dice(y_true, y_pred):
    # computes the dice score on two tensors

    sum_p = K.sum(y_pred, axis=0)
    sum_r = K.sum(y_true, axis=0)
    sum_pr = K.sum(y_true * y_pred, axis=0)
    dice_numerator = 2 * sum_pr
    dice_denominator = sum_r + sum_p
    dice_score = (dice_numerator + K.epsilon()) / (dice_denominator + K.epsilon())
    return dice_score


def dice_whole_metric(y_true, y_pred):
    # computes the dice for the whole tumor

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_whole = K.sum(y_true_f[..., 1:], axis=1)
    p_whole = K.sum(y_pred_f[..., 1:], axis=1)
    dice_whole = dice(y_whole, p_whole)
    return dice_whole


def dice_en_metric(y_true, y_pred):
    # computes the dice for the enhancing region

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))
    y_enh = y_true_f[:, -1]
    p_enh = y_pred_f[:, -1]
    dice_en = dice(y_enh, p_enh)
    return dice_en


def dice_core_metric(y_true, y_pred):
    ##computes the dice for the core region

    y_true_f = K.reshape(y_true, shape=(-1, 4))
    y_pred_f = K.reshape(y_pred, shape=(-1, 4))

    # workaround for tf
    # y_core=K.sum(tf.gather(y_true_f, [1,3],axis =1),axis=1)
    # p_core=K.sum(tf.gather(y_pred_f, [1,3],axis =1),axis=1)

    y_core = K.sum(y_true_f[:, 2:], axis=1)
    p_core = K.sum(y_pred_f[:, 2:], axis=1)
    dice_core = dice(y_core, p_core)
    return dice_core

def gen_dice_score(y_true, y_pred):
  y_true_f = K.reshape(y_true,shape=(-1,4))
  y_pred_f = K.reshape(y_pred,shape=(-1,4))
  sum_p=K.sum(y_pred_f,axis=-2)
  sum_r=K.sum(y_true_f,axis=-2)
  sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
  weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
  generalised_dice_numerator =2*K.sum(weights*sum_pr)
  generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
  generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
  return generalised_dice_score


def gen_dice_loss(y_true, y_pred):
  return 1 - gen_dice_score(y_true, y_pred)


# model_path=config.MODEL_PATH

# Handle compatibility issue with newer TensorFlow versions
try:
    model = tf.keras.models.load_model('finalvalaug.h5', custom_objects={'gen_dice_loss': gen_dice_loss, 'dice_whole_metric':dice_whole_metric, 'dice_en_metric': dice_en_metric, 'dice_core_metric': dice_core_metric}, compile=False)
except (ValueError, TypeError) as e:
    if 'groups' in str(e) or 'Conv3DTranspose' in str(e):
        # Workaround for compatibility: load model weights only
        from Model import Unet_3d
        # Build model architecture first with input shape (160, 192, 128, 4)
        input_img = tf.keras.layers.Input(shape=(160, 192, 128, 4))
        model = Unet_3d(input_img, n_filters=8, dropout=0.2, batch_norm=True)
        # Load weights
        model.load_weights('finalvalaug.h5')
        print("Model loaded with compatibility workaround")
    else:
        raise

"""### Prediction """
def _to_channels_last4(arr):
    # Convert various NIfTI array shapes to (X, Y, Z, 4)
    a = np.asarray(arr)
    # First squeeze singleton dims where safe
    if a.ndim > 5:
        a = np.squeeze(a)
    if a.ndim == 5:
        a = np.squeeze(a)
    # If 3D, repeat across channels
    if a.ndim == 3:
        return np.stack([a, a, a, a], axis=-1)
    # After squeeze, handle 4D cases
    if a.ndim == 4:
        if a.shape[-1] == 4:
            return a
        if 4 in a.shape:
            ch_axis = list(a.shape).index(4)
            return np.moveaxis(a, ch_axis, -1)
    raise ValueError(f"Unsupported NIfTI shape {a.shape}; must include a 4-length channel axis")



def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzeor region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """

    pixels = volume[volume > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (volume - mean) / std
    return out


def normalize(image):
    img1 = itensity_normalize_one_volume(image[..., 0])
    img2 = itensity_normalize_one_volume(image[..., 1])
    img3 = itensity_normalize_one_volume(image[..., 2])
    img4 = itensity_normalize_one_volume(image[..., 3])
    img = np.stack((img1, img2, img3, img4), axis=-1)
    return img


def input_image(image):
    image_path = os.path.join(config.IMAGES_DATA_DIR, image)
    img = nib.load(image_path)
    image_data = img.dataobj
    image_data = _to_channels_last4(image_data)

    image_data = image_data[34:194, 22:214, 13:141, ]
    image_data = normalize(image_data)
    # Reshaping the Input Image and Ground Truth(Mask)
    reshaped_image_data=image_data.reshape(1,160,192,128,4)

    print(reshaped_image_data.shape)
    print(type(reshaped_image_data))

    # Prediction - Our Segmentation
    Y_hat = model.predict(x=reshaped_image_data)
    Y_hat = np.argmax(Y_hat, axis=-1)
    print(f"Y_hat shape - {Y_hat.shape}")

    # Read the Input Image and Predicted Mask
    image = reshaped_image_data[0, :, :, :, 0].T
    mask = Y_hat[0].T

    # Quantification and visualization
    metrics = compute_tissue_quantification(mask)
    overlay_fig = overlay_segmentation(image, mask)
    bar_fig = bar_chart_from_metrics(metrics)
    pie_fig = pie_chart_from_metrics(metrics)

    # Save CSV metrics with case id derived from filename
    case_id = image_path.split(os.sep)[-1]
    if case_id.endswith('.nii.gz'):
        case_id = case_id[:-7]
    elif case_id.endswith('.nii'):
        case_id = case_id[:-4]
    save_quantification_csv(metrics, os.path.join('BrainTumorData', 'quant_metrics.csv'), {"case_id": case_id})

    # For Colorscale
    pl_bone=[[0.0, 'rgb(0, 0, 0)'],
             [0.05, 'rgb(10, 10, 14)'],
             [0.1, 'rgb(21, 21, 30)'],
             [0.15, 'rgb(33, 33, 46)'],
             [0.2, 'rgb(44, 44, 62)'],
             [0.25, 'rgb(56, 55, 77)'],
             [0.3, 'rgb(66, 66, 92)'],
             [0.35, 'rgb(77, 77, 108)'],
             [0.4, 'rgb(89, 92, 121)'],
             [0.45, 'rgb(100, 107, 132)'],
             [0.5, 'rgb(112, 123, 143)'],
             [0.55, 'rgb(122, 137, 154)'],
             [0.6, 'rgb(133, 153, 165)'],
             [0.65, 'rgb(145, 169, 177)'],
             [0.7, 'rgb(156, 184, 188)'],
             [0.75, 'rgb(168, 199, 199)'],
             [0.8, 'rgb(185, 210, 210)'],
             [0.85, 'rgb(203, 221, 221)'],
             [0.9, 'rgb(220, 233, 233)'],
             [0.95, 'rgb(238, 244, 244)'],
             [1.0, 'rgb(255, 255, 255)']]

    r,c = image[0].shape
    n_slices = image.shape[0]
    height = (image.shape[0]-1) / 10
    grid = np.linspace(0, height, n_slices)
    slice_step = grid[1] - grid[0]

    rm,cm = mask[0].shape
    nm_slices = mask.shape[0]
    height_m = (mask.shape[0]-1) / 10
    grid_m = np.linspace(0, height_m, nm_slices)
    slice_step_m = grid_m[1] - grid_m[0]

    initial_slice = go.Surface(
                         z=height*np.ones((r,c)),
                         surfacecolor=np.flipud(image[-1]),
                         colorscale=pl_bone,
                         showscale=False)

    initial_slice_m = go.Surface(
                         z=height_m*np.ones((rm,cm)),
                         surfacecolor=np.flipud(mask[-1]),
                         colorscale=pl_bone,
                         showscale=False)

    frames = [go.Frame(data=[dict(type='surface',
                              z=(height-k*slice_step)*np.ones((r,c)),
                              surfacecolor=np.flipud(image[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, n_slices)]

    frames_m = [go.Frame(data=[dict(type='surface',
                              z=(height_m-k*slice_step_m)*np.ones((rm,cm)),
                              surfacecolor=np.flipud(mask[-1-k]))],
                              name=f'frame{k+1}') for k in range(1, nm_slices)]

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [dict(steps = [dict(method= 'animate',
                                  args= [[f'frame{k+1}'],
                                        dict(mode= 'immediate', frame= dict(duration=20, redraw= True),transition=dict(duration= 0))
                                        ],
                                  label=f'{k+1}'
                                  )for k in range(n_slices)],
                    active=17,
                    transition= dict(duration= 0),
                    x=0, # slider starting position
                    y=0,
                    currentvalue=dict(font=dict(size=12),
                                      prefix='slice: ',
                                      visible=True,
                                      xanchor= 'center'
                                     ),
                   len=1.0) #slider length
               ]

    layout3d = dict(title_text='Slices of Brain in volumetric data: Input Image', title_x=0.5,
                    template="plotly_dark",
                    width=600,
                    height=600,
                    scene_zaxis_range=[-0.1, 12.8],
                    updatemenus = [
                        {
                            "buttons": [
                                {
                                    "args": [None, frame_args(50)],
                                    "label": "&#9654;", # play symbol
                                    "method": "animate",
                                },
                                {
                                    "args": [[None], frame_args(0)],
                                    "label": "&#9724;", # pause symbol
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 0, "t": 60},
                            "type": "buttons",
                            "x": 0,
                            "y": 0,
                        }
                     ],
                     sliders=sliders
                )

    layout3d_m = dict(title_text='Slices of Mask: Brain Segmentation', title_x=0.5,
                    template="plotly_dark",
                    width=600,
                    height=600,
                    scene_zaxis_range=[-0.1, 12.8],
                    updatemenus = [
                        {
                            "buttons": [
                                {
                                    "args": [None, frame_args(50)],
                                    "label": "&#9654;", # play symbol
                                    "method": "animate",
                                },
                                {
                                    "args": [[None], frame_args(0)],
                                    "label": "&#9724;", # pause symbol
                                    "method": "animate",
                                },
                            ],
                            "direction": "left",
                            "pad": {"r": 0, "t": 60},
                            "type": "buttons",
                            "x": 0,
                            "y": 0,
                        }
                     ],
                     sliders=sliders
                )

    fig1 = go.Figure(data=[initial_slice], layout=layout3d, frames=frames)
    fig2 = go.Figure(data=[initial_slice_m], layout=layout3d_m, frames=frames_m)

    return fig1, fig2, overlay_fig, bar_fig, pie_fig, metrics

# Custom CSS for modern UI
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    color: #e0e0e0;
}

.navbar {
    background: rgba(26, 31, 58, 0.95);
    backdrop-filter: blur(10px);
    padding: 1rem 2rem;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
    position: sticky;
    top: 0;
    z-index: 1000;
    border-bottom: 1px solid rgba(79, 172, 254, 0.2);
}

.navbar-content {
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-brand {
    font-size: 1.5rem;
    font-weight: 700;
    color: #4FACFE;
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.navbar-brand::before {
    content: '🧠';
    font-size: 1.8rem;
}

.navbar-nav {
    display: flex;
    gap: 1.5rem;
    list-style: none;
}

.nav-link {
    color: #b0b0b0;
    text-decoration: none;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    transition: all 0.3s ease;
}

.nav-link:hover {
    color: #4FACFE;
    background: rgba(79, 172, 254, 0.1);
}

.nav-link.active {
    color: #4FACFE;
    background: rgba(79, 172, 254, 0.15);
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem;
}

.page-header {
    text-align: center;
    margin-bottom: 3rem;
    padding: 2rem 0;
}

.page-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.page-subtitle {
    font-size: 1.1rem;
    color: #a0a0a0;
    font-weight: 400;
}

.card {
    background: rgba(26, 31, 58, 0.6);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(79, 172, 254, 0.1);
    transition: all 0.3s ease;
}

.card:hover {
    border-color: rgba(79, 172, 254, 0.3);
    box-shadow: 0 6px 30px rgba(79, 172, 254, 0.2);
}

.card-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #4FACFE;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.1) 100%);
    border-radius: 10px;
    padding: 1.5rem;
    border: 1px solid rgba(79, 172, 254, 0.2);
}

.stat-label {
    font-size: 0.9rem;
    color: #a0a0a0;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #4FACFE;
}

.btn {
    padding: 0.75rem 2rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    text-decoration: none;
    display: inline-block;
    text-align: center;
}

.btn-primary {
    background: linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%);
    color: white;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 172, 254, 0.6);
}

.btn-secondary {
    background: rgba(79, 172, 254, 0.1);
    color: #4FACFE;
    border: 1px solid rgba(79, 172, 254, 0.3);
}

.btn-secondary:hover {
    background: rgba(79, 172, 254, 0.2);
}

.upload-area {
    border: 2px dashed rgba(79, 172, 254, 0.4);
    border-radius: 12px;
    padding: 3rem 2rem;
    text-align: center;
    background: rgba(26, 31, 58, 0.4);
    transition: all 0.3s ease;
    cursor: pointer;
    margin: 2rem 0;
}

.upload-area:hover {
    border-color: rgba(79, 172, 254, 0.7);
    background: rgba(26, 31, 58, 0.6);
}

.upload-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}

.upload-text {
    font-size: 1.2rem;
    color: #e0e0e0;
    margin-bottom: 0.5rem;
}

.upload-hint {
    font-size: 0.9rem;
    color: #a0a0a0;
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(79, 172, 254, 0.3);
    border-radius: 50%;
    border-top-color: #4FACFE;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    margin: 1rem 0;
    padding: 1rem;
    background: rgba(26, 31, 58, 0.4);
    border-radius: 8px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.legend-color {
    width: 20px;
    height: 20px;
    border-radius: 4px;
}

.tooltip-icon {
    display: inline-block;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: rgba(79, 172, 254, 0.2);
    color: #4FACFE;
    text-align: center;
    line-height: 18px;
    font-size: 12px;
    cursor: help;
    margin-left: 0.5rem;
}

.graph-container {
    background: rgba(26, 31, 58, 0.4);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1.5rem;
}

@media (max-width: 768px) {
    .page-title {
        font-size: 2rem;
    }
    
    .stats-grid {
        grid-template-columns: 1fr;
    }
    
    .navbar-content {
        flex-direction: column;
        gap: 1rem;
    }
}
"""

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
        'rel': 'stylesheet'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Brain Tumor Segmentation</title>
        {%favicon%}
        {%css%}
        <style>''' + custom_css + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

colors = {
    'background': '#0a0e27',
    'card': '#1a1f3a',
    'primary': '#4FACFE',
    'secondary': '#00F2FE',
    'text': '#e0e0e0',
    'text-muted': '#a0a0a0'
}
# Helper function to create navigation bar
def create_navbar(current_path='/'):
    return html.Nav(className="navbar", children=[
        html.Div(className="navbar-content", children=[
            html.A("Brain Tumor Segmentation", href="/", className="navbar-brand"),
            html.Ul(className="navbar-nav", children=[
                html.Li(html.A("Home", href="/", className="nav-link" + (" active" if current_path == "/" else ""))),
                html.Li(html.A("Upload Scan", href="/upload", className="nav-link" + (" active" if current_path == "/upload" else ""))),
            ])
        ])
    ])

# Helper function to create stat cards
def create_stat_card(label, value, color="#4FACFE"):
    return html.Div(className="stat-card", children=[
        html.Div(className="stat-label", children=label),
        html.Div(className="stat-value", style={"color": color}, children=f"{value:.2f}%")
    ])

# Helper function to create legend
def create_legend():
    return html.Div(className="legend", children=[
        html.Div(className="legend-item", children=[
            html.Div(className="legend-color", style={"background": "cyan"}),
            html.Span("Edema", style={"color": "#e0e0e0"})
        ]),
        html.Div(className="legend-item", children=[
            html.Div(className="legend-color", style={"background": "yellow"}),
            html.Span("Necrotic Core", style={"color": "#e0e0e0"})
        ]),
        html.Div(className="legend-item", children=[
            html.Div(className="legend-color", style={"background": "red"}),
            html.Span("Enhancing Tumor", style={"color": "#e0e0e0"})
        ]),
    ])

# Load initial data
fig_1, fig_2, fig_overlay, fig_bar, _, initial_metrics = input_image("test4d.nii.gz")

# Home page layout
index_page = html.Div([
    create_navbar('/'),
    html.Div(className="container", children=[
        html.Div(className="page-header", children=[
            html.H1("Brain Tumor Segmentation", className="page-title"),
            html.P("AI-Powered 3D Brain MRI Analysis & Tumor Detection", className="page-subtitle")
        ]),
        
        html.Div(className="card", children=[
            html.Div(className="card-header", children=["📊 Summary Statistics"]),
            html.Div(className="stats-grid", children=[
                create_stat_card("Edema", initial_metrics.get("Edema", 0.0), "#00F2FE"),
                create_stat_card("Necrotic Core", initial_metrics.get("Necrotic core", 0.0), "#FFD700"),
                create_stat_card("Enhancing Tumor", initial_metrics.get("Enhancing tumor", 0.0), "#FF4444"),
                create_stat_card("Total Lesion Load", initial_metrics.get("Total lesion load", 0.0), "#4FACFE"),
            ]),
            create_legend()
        ]),
        
        html.Div(className="card", style={"textAlign": "center", "padding": "2rem"}, children=[
            html.P("Upload a new brain MRI scan to analyze tumor segmentation", style={"marginBottom": "1.5rem", "fontSize": "1.1rem"}),
            dcc.Link(html.Button("Upload Brain Scan", className="btn btn-primary"), href='/upload')
        ]),
        
        html.Div(className="row", children=[
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["🧠 Input MRI Scan"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g1', figure=fig_1, config={'displayModeBar': True})
                    ])
                ])
            ]),
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["🎯 Segmentation Result"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g2', figure=fig_2, config={'displayModeBar': True})
                    ])
                ])
            ]),
        ]),
        
        html.Div(className="row", children=[
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["🔍 Overlay View"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g_overlay', figure=fig_overlay, config={'displayModeBar': True})
                    ])
                ])
            ]),
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["📊 Quantification"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g_bar', figure=fig_bar, config={'displayModeBar': True})
                    ])
                ])
            ]),
        ])
    ])
], style={'backgroundColor': colors['background'], 'minHeight': '100vh'})

# Upload page layout
page_1_layout = html.Div([
    create_navbar('/upload'),
    html.Div(className="container", children=[
        html.Div(className="page-header", children=[
            html.H1("Upload Brain Scan", className="page-title"),
            html.P("Upload a NIfTI (.nii or .nii.gz) brain MRI scan for analysis", className="page-subtitle")
        ]),
        
        html.Div(className="card", children=[
            html.Div(className="card-header", children=["📤 Upload MRI Scan"]),
            dcc.Upload(
                id='upload-image',
                children=html.Div(className="upload-area", children=[
                    html.Div(className="upload-icon", children="📁"),
                    html.Div(className="upload-text", children="Drag and Drop your NIfTI file here"),
                    html.Div(className="upload-hint", children="or click to select a file (.nii or .nii.gz)"),
                ]),
                style={'width': '100%'},
                multiple=True
            ),
            html.Div(id='upload-status', style={'marginTop': '1rem'}),
        ]),
        
        html.Div(id='output-image-upload'),
        
        html.Div(style={'textAlign': 'center', 'marginTop': '2rem'}, children=[
            dcc.Link(html.Button("← Back to Home", className="btn btn-secondary"), href='/')
        ])
    ])
], style={'backgroundColor': colors['background'], 'minHeight': '100vh'}),

def parse_contents(filename_only):
    img, msk, overlay_fig, bar_fig, _, metrics = input_image(filename_only)
    
    return html.Div([
        html.Div(className="card", children=[
            html.Div(className="card-header", children=["📊 Analysis Results"]),
            html.Div(className="stats-grid", children=[
                create_stat_card("Edema", metrics.get("Edema", 0.0), "#00F2FE"),
                create_stat_card("Necrotic Core", metrics.get("Necrotic core", 0.0), "#FFD700"),
                create_stat_card("Enhancing Tumor", metrics.get("Enhancing tumor", 0.0), "#FF4444"),
                create_stat_card("Total Lesion Load", metrics.get("Total lesion load", 0.0), "#4FACFE"),
            ]),
            create_legend()
        ]),
        
        html.Div(className="row", children=[
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["🧠 Input MRI Scan"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g1', figure=img, config={'displayModeBar': True})
                    ])
                ])
            ]),
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["🎯 Segmentation Result"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g2', figure=msk, config={'displayModeBar': True})
                    ])
                ])
            ]),
        ]),
        
        html.Div(className="row", children=[
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["🔍 Overlay View"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g_overlay', figure=overlay_fig, config={'displayModeBar': True})
                    ])
                ])
            ]),
            html.Div(className="six columns", children=[
                html.Div(className="card", children=[
                    html.Div(className="card-header", children=["📊 Quantification"]),
                    html.Div(className="graph-container", children=[
                        dcc.Graph(id='g_bar', figure=bar_fig, config={'displayModeBar': True})
                    ])
                ])
            ]),
        ])
    ])


@app.callback([Output('output-image-upload', 'children'),
               Output('upload-status', 'children')],
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename')])

def update_output(image, filenames):
    if not image:
        return None, html.Div()

    saved_names = []
    status_messages = []
    
    for i, image_str in enumerate(image):
        try:
            data = image_str.encode("utf8").split(b";base64,")[1]
            orig_name = filenames[i] if filenames and i < len(filenames) else f"image_{i+1}.nii.gz"
            
            if not (orig_name.endswith('.nii') or orig_name.endswith('.nii.gz')):
                status_messages.append(html.Div([
                    html.Span("❌ ", style={"color": "#FF4444"}),
                    html.Span(f"{orig_name}: Invalid file type. Please upload a NIfTI file (.nii or .nii.gz)", 
                             style={"color": "#FF4444"})
                ], style={"padding": "0.5rem", "margin": "0.5rem 0", "background": "rgba(255, 68, 68, 0.1)", 
                         "borderRadius": "6px"}))
                continue

            # preserve extension
            ext = '.nii.gz' if orig_name.endswith('.nii.gz') else '.nii'
            save_name = f"upload_{i+1}{ext}"
            save_path = os.path.join(config.IMAGES_DATA_DIR, save_name)
            
            with open(save_path, "wb") as fp:
                fp.write(base64.decodebytes(data))

            # Validate NIfTI shape: 4D with 4 channels
            img = nib.load(save_path)
            # Normalize to channels-last 4D; accepts 3D and converts by repeating across 4 channels
            _ = _to_channels_last4(img.dataobj)

            saved_names.append(save_name)
            status_messages.append(html.Div([
                html.Span("✅ ", style={"color": "#00F2FE"}),
                html.Span(f"{orig_name}: Uploaded successfully", style={"color": "#00F2FE"})
            ], style={"padding": "0.5rem", "margin": "0.5rem 0", "background": "rgba(0, 242, 254, 0.1)", 
                     "borderRadius": "6px"}))
            
        except Exception as e:
            status_messages.append(html.Div([
                html.Span("❌ ", style={"color": "#FF4444"}),
                html.Span(f"Error processing {orig_name if filenames and i < len(filenames) else 'file'}: {str(e)}", 
                         style={"color": "#FF4444"})
            ], style={"padding": "0.5rem", "margin": "0.5rem 0", "background": "rgba(255, 68, 68, 0.1)", 
                     "borderRadius": "6px"}))

    if saved_names:
        # Show processing message
        processing_msg = html.Div([
            html.Div(className="loading-spinner", style={"display": "inline-block", "marginRight": "0.5rem"}),
            html.Span("Processing scan... This may take a moment.", style={"color": "#4FACFE", "fontWeight": "500"})
        ], style={"padding": "1rem", "margin": "1rem 0", "background": "rgba(79, 172, 254, 0.1)", 
                 "borderRadius": "8px", "textAlign": "center"})
        
        # Display only the first upload
        children = [processing_msg, parse_contents(saved_names[0])]
        return children, html.Div(status_messages)
    
    return None, html.Div(status_messages)

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/upload':
        return page_1_layout
    else:
        return index_page

if __name__ == '__main__':
    app.run(debug=True, port=8051)
