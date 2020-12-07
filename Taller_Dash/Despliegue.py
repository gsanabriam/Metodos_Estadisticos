import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  ## Support vector classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier ### RandomForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, classification_report
from sklearn.svm import SVC  ## Support vector classifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, tree
from sklearn.model_selection import GridSearchCV
import pickle
from skimage import io
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
Modelo=['Árbol Desición','Random Forest','Regresión Lógistica']

app = dash.Dash(external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        html.Div([
            html.H2('Pronóstico de vida para pacientes con problemas cardiovasculares', style={'textAlign': 'center', 'color': 'black'}),
            html.Table([
                html.Tr([
                    html.Tr([
                        html.H4('Ingresar características de la sangre'),
                    ]),
                    html.Tr([
                        html.Td('Edad:'),
                        html.Td([dcc.Input(id="input-1", type="number", placeholder='Age',min=0, max=100)]),
                        html.Td(),
                        html.Td('Anemia:'),
                        html.Td([dcc.Input(id="input-2", type="number", placeholder='Anemia')]),
                    ]),
                    html.Tr([
                        html.Td('Creatina:'),
                        html.Td([dcc.Input(id="input-3", type="number", placeholder='Creatina')]),
                        html.Td(),
                        html.Td('Diabetes:'),
                        html.Td([dcc.Input(id="input-4", type="number", placeholder='Diabetes')]),
                    ]),
                    html.Tr([
                        html.Td('Ejection Fraction:'),
                        html.Td([dcc.Input(id="input-5", type="number", placeholder='Ejection Fraction')]),
                        html.Td(),
                        html.Td('High blood pressure:'),
                        html.Td([dcc.Input(id="input-6", type="number", placeholder='High blood pressure')]),
                    ]),
                    html.Tr([
                        html.Td('Platelets:'),
                        html.Td([dcc.Input(id="input-7", type="number", placeholder='Platelets')]),
                        html.Td(),
                        html.Td('Serum creatinine:'),
                        html.Td([dcc.Input(id="input-8", type="number", placeholder='Serum creatinine')]),

                    ]),
                    html.Tr([
                        html.Td('Serum sodium:'),
                        html.Td([dcc.Input(id="input-9", type="number", placeholder='Serum sodium')]),
                        html.Td(),
                        html.Td('Sex:'),
                        html.Td([dcc.Input(id="input-10", type="text", placeholder="Sex")]),
                    ]),
                    html.Tr([
                        html.Td('Smoking:'),
                        html.Td([dcc.Input(id="input-11", type="number", placeholder='Smoking', min=0,max=1)]),
                        html.Td(),
                        html.Td('Time:'),
                        html.Td([dcc.Input(id="input-12", type="number", placeholder='Time')]),

                    ]), 
                ]),
                 
            ],style={"display": "inline-block", 'width': '100%', "border-style": "dashed", "border-width": "1px", "color": "black","margin-bottom":"0px"}),
        ]),
        
        html.Br(),
        html.Div([
             html.H4( "Seleccione Modelo:"),
             dcc.Dropdown(
                  id='Model',
                   options=[{'label': i, 'value': i} for i in Modelo],
                   value=None,),
                    
        ],style={"display": "inline-block", 'width': '100%', "border-style": "dashed", "border-width": "1px", "color": "black" }),
        
        html.Div([
            html.Table([
               html.Tr([
                   html.Td([
                       html.Tr([
                          html.H2(id='Etq'),   
                       ]),
                       html.Tr([
                          dcc.Graph(id='f1'),           
                       ]),                      
                       
                   ]),
                   html.Td([
                       html.Tr([
                            html.H3(id='Etq2'),
                       ]),
                       html.Tr([
                            dcc.Graph(id='f2'),   
                       ]),
  
                       
                   ]),
                    
                ],style={"display": "inline-block", 'width': '100%', "border-style": "dashed", "border-width": "1px", "color": "black" }),

            ]),

                        
        ]),

    ])


@app.callback(
              Output('f1', 'figure'),
              Output('f2', 'figure'),
              Output('Etq','children'),
              Output('Etq2','children'),
              Input('input-1', 'value'),
              Input('input-2', 'value'),
              Input('input-3', 'value'),
              Input('input-4', 'value'),
              Input('input-5', 'value'),
              Input('input-6', 'value'),
              Input('input-7', 'value'),
              Input('input-8', 'value'),
              Input('input-9', 'value'),
              Input('input-10', 'value'),
              Input('input-11', 'value'),
              Input('input-12', 'value'),
              Input('Model','value')
)
def update_output(input1, input2,input3, input4,input5,input6,input7,input8,input9,input10,input11,input12,Model):
    
    
    #Rutas 
    

    #Valida genero
    if (input10=='Hombre'):
        sex=1
    else:
        sex=0
        
        
    #Crea dataframe
    undato=pd.DataFrame({'age':[input1],
                    'anaemia':[input2],
                    'creatinine_phosphokinase':[input3],
                    'diabetes':[input4],
                    'ejection_fraction':[input5],
                    'high_blood_pressure':[input6],
                    'platelets':[input7],
                    'serum_creatinine':[input8],
                    'serum_sodium':[input9],
                    'sex':[sex],
                    'smoking':[input11],
                    'time':[input12]

                   })
    
    #Definición del modelo
    if(Model!=None):
        if(Model=='Árbol Desición'):
            filename = 'ModelAbrol.sav'
            Etiqueta='Resultado Árbol'
        elif (Model=='Random Forest'):
            filename= 'ModelRF.sav'   
            Etiqueta='Resultado Random forest'
        else:
            filename= 'ModelRL.sav'
            Etiqueta='Resultado Regresión lógistica'

        
    loaded_model = pickle.load(open(filename, 'rb'))
    pro=loaded_model.predict_proba(undato[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']])
    
    result = loaded_model.predict(undato[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']])

    Dato=pd.DataFrame({'Estado':['Vivo','Muerto'],
                               'Porcentaje':[pro[0][0],pro[0][1]]
                              })
                           
                           
    d=Dato[Dato['Estado']=='Vivo']
    valor=d.iloc[0,1]
    Etiqueta2='Tiene el {:2.2%} de seguir con vida por sus caracteristicas sanguíneas'.format( valor)
    
    if(valor>=0.5):
        fig2 = px.imshow(io.imread('https://github.com/gsanabriam/Metodos_Estadisticos/blob/main/Imagen_alentado.png?raw=true'))
        
    else:
        fig2 = px.imshow(io.imread('https://github.com/gsanabriam/Metodos_Estadisticos/blob/main/Imagen_enfermo.png?raw=true'))



    fig = px.pie(Dato, values='Porcentaje', names='Estado', title='Porcentaje de vida')
    
    
    return fig, fig2, Etiqueta, Etiqueta2


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)
