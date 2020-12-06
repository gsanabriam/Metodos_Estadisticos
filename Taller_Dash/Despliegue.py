#Librerias
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
from skimage import io
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


#Selecciona Modelo
Modelo=['Árbol Desición','Random Forest','Regresión Lógistica']
app = dash.Dash()

app.layout = html.Div(
    [
        html.Div([
            html.H2('Pronóstico de vida para pacientes con problemas cardiovasculares', style={'textAlign': 'center', 'color': 'black'}),
  
        
        ]),
        html.Div([
            html.Table([
                    dcc.Input(id="input-5", type="number", placeholder='Ejection Fraction'),
                    dcc.Graph(id='f1'),                       
             ],style={"display": "inline-block", 'width': '100%', "border-style": "dashed", "border-width": "1px", "color": "black" }),


                        
        ]),

    ])


@app.callback(
              Output('f1', 'figure'),
              Input('input-5','value')
)
def update_output(f):
    if(f==1):
        fig = px.imshow(io.imread('https://github.com/gsanabriam/Metodos_Estadisticos/blob/main/Imagen_alentado.png?raw=true'))
        return fig



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)
