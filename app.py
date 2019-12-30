# Developed By Code|<Ill at 6/25/2019
# Developed VM IP 203.241.246.158

#Scikit LEarn and Pandas Imports
import pandas as pd
import pickle
from sklearn.metrics import classification_report, average_precision_score, mean_squared_error
from sklearn.metrics import confusion_matrix

#Dash Dependencies
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import dash_table

#Plotly Imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
plotly.tools.set_credentials_file(username='c.sabyasachi99', api_key='y5FSl1jIheriCgKbK3Ff')

#Self Declared Modules and Packages
from Utilities import helper_functions
from Utilities import filtering
from Utilities import feature_engineering
from Model import Model



# External CSS
external_stylesheets=["assets/template.css",
                      "assets/bootstrap.min.css"]
# External Scripts
external_scripts=["assets/bootstrap.min.js"
                  ]

# Initializing the Default Constructor of Dash Framework and the Application
app=dash.Dash(__name__,
              external_scripts=external_scripts,
              external_stylesheets=external_stylesheets)


app.layout=html.Div([
    #Banner
    html.Div(className='header', children=[
        html.Div(className='header-content', children=[
            html.H2("Pet Activity Detection", id='title'),
            html.Img(src="https://i.ibb.co/f4HBQmv/Inje-1.png",id='inje_logo'),
            html.Img(src="https://i.ibb.co/f4HBQmv/Inje-1.png",id='ida_logo')
        ])
    ]),

    #Body
    html.Div(className='body',children=[
        html.Div(className='section-1',children=[
            html.Div(className='container', children=[
                html.Div(className='row', children=[
                    html.Div(className='col-lg-3 white-bg', children=[
                        html.H2("Select Activity Data", id='sub-title',className='selector'),
                        dcc.Dropdown(
                            id='Dataset',
                            options=[
                                {'label':'Walking','value':'Test_Walk'},
                                {'label':'Sitting','value':'Test_Sit'},
                                {'label':'Stay','value':'Test_Stay'},
                                {'label':'Eat','value':'Test_Eat'},
                                {'label':'Sideway','value':'Test_Sideway'},
                                {'label':'Nosework','value':'Test_Nosework'},
                                {'label':'Jump','value':'Test_Jump'}
                            ],
                            multi=True,
                            value=['Test_Walk','Test_Sit']
                        ),
                        html.Div(className='button-holder',children=[
                            html.Button('Merge Data', id='click_button',className='button')
                        ]),
                        html.H4("Model Initialization", className='selector'),
                        html.Div(className='button-holder',children=[
                            html.Button('Run Model', id='click_button_1',className='button')
                        ])


                    ]),

                    html.Div(className='col-lg-3 white-bg',children=[
                        html.H4("BioSignal View Selection",className='selector'),
                        dcc.RadioItems(
                            id='view_choice',
                            options=[
                                {'label':'Neck Accelerometer', 'value':'N-Acc'},
                                {'label':'Neck Gyroscope', 'value':'N-Gyro'},
                                {'label':'Tail Accelerometer', 'value':'T-Acc'},
                                {'label':'Tail Gyroscope', 'value':'T-Gyro'}
                            ],
                            value='N-Acc',
                            className='radio_buttons'
                        ),
                        html.Div(className='note-tag', children=[
                            dcc.Markdown(children='''
                            >**Note :** The Radio Items provides multiple view of the signals.''')
                        ],style={'margin-top':'12px'})

                    ]),

                    html.Div(className='col-lg-6 white-bg graph', children=[
                        html.H4("Visual Representation of Data", className='selector'),
                        dcc.Graph(id='axis_area',style={'width':'98%','float':'right', 'height':'300px','text-align':'center','position':'relative'})
                    ])
                ])
            ])
        ]),
############################################################################################ Section 2 ############################################################################################################
        html.Div(className='section-2',children=[
            html.Div(className='container',children=[
                html.Div(className='row', children=[
                    html.Div(className='col-lg-4 white-bg', children=[
                        html.H4("Data Table", className='selector'),
                        html.Div(id='Output-Data-Table'),
                        #dcc.Graph(id='table_area',style={'width': '98%', 'float': 'right', 'height': '300px', 'text-align': 'center','position': 'relative'})
                    ]),
                    html.Div(className='col-lg-4 white-bg', children=[
                        html.H4("Testing Distribution", className='selector'),
                        dcc.Graph(id='dist_area',style={'width': '98%', 'float': 'right', 'height': '300px', 'text-align': 'center','position': 'relative'})
                    ]),
                    html.Div(className='col-lg-4 white-bg', children=[
                        html.H4("3D Distribution", className='selector'),
                        dcc.Graph(id='3d_area',style={'width': '98%', 'float': 'right', 'height': '300px', 'text-align': 'center','position': 'relative'})
                    ])
                ])
            ])
        ]),
##################################################################################### Section 3 - Modelling ####################################################################################################
        html.Div(className='section-3',children=[
            html.Div(className='container',children=[
                html.Div(className='row', children=[
                    html.Div(className='col-lg-12 white-bg', children=[
                        html.H4("Model Performance and Evaluation", className='selector'),
                        html.Div(id='eval_section')
                        ])
                    ])
                ])
            ])
        ])


    ],style={'background-color':'#f5f5f5'})


################################################################################## Call Back Functions #######################################################################################################
## 1. Call Back For Visual Representation of Data
@app.callback(
    Output('axis_area','figure'),
    [Input('Dataset','value'),
    Input('view_choice','value')])

def update_graph(data,view):
    data_dir='Data/'

    if not data:
        return {
            'data': [],
            'layout': go.Layout(
                xaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                },

                yaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                }
            )
        }

    else:

        data_list=[]

        for i in data:
            dataset_loop=pd.read_csv(data_dir+i+'.csv',sep=',',encoding="utf-8",error_bad_lines=False,low_memory = False) #Reading Test Datasets
            data_list.append(dataset_loop) #Data Objects
        main_data=pd.concat(data_list, axis=0, ignore_index=True)

        #Choosing The View of That need to be Shown to the USer
        if(view=='N-Acc'):
            #Filtering The Signals
            main_data['N-AccX'] = helper_functions.butter_lowpass_filter(main_data['N-AccX'], cutoff=3.667, fs=33.3, order=5)
            main_data['N-AccY'] = helper_functions.butter_lowpass_filter(main_data['N-AccY'], cutoff=3.667, fs=33.3, order=5)
            main_data['N-AccZ'] = helper_functions.butter_lowpass_filter(main_data['N-AccZ'], cutoff=3.667, fs=33.3, order=5)

            # Resultant Calculation
            resultant_value=helper_functions.resultant(main_data['N-AccX'],main_data['N-AccY'],main_data['N-AccZ'])

            return {
                'data':[go.Scatter(
                    x=list(range(len(list(resultant_value)))),
                    y=list(resultant_value),
                    mode='lines',
                    line = dict(
                    color = '#03B5AA',
                    width = 2)
                )],
                'layout': go.Layout(
                    xaxis={'title':'Time Epoch'},
                    yaxis={'title':'Value'},
                    showlegend=False,
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                    hovermode='closest'
                )
            }
        elif(view=='N-Gyro'):
            # Filtering The Signals
            main_data['N-GyroX'] = helper_functions.butter_lowpass_filter(main_data['N-GyroX'], cutoff=3.667, fs=33.3,order=5)
            main_data['N-GyroY'] = helper_functions.butter_lowpass_filter(main_data['N-GyroY'], cutoff=3.667, fs=33.3,order=5)
            main_data['N-GyroZ'] = helper_functions.butter_lowpass_filter(main_data['N-GyroZ'], cutoff=3.667, fs=33.3,order=5)

            #Resultant Calculation
            resultant_value = helper_functions.resultant(main_data['N-GyroX'], main_data['N-GyroY'], main_data['N-GyroZ'])
            return {
                'data':[go.Scatter(
                    x=list(range(len(list(resultant_value)))),
                    y=list(resultant_value),
                    mode='lines',
                    line = dict(
                    color = '#D1495B',
                    width = 2)
                )],
                'layout': go.Layout(
                    xaxis={'title':'Time Epoch'},
                    yaxis={'title':'Value'},
                    showlegend=False,
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                    hovermode='closest'
                )
            }
        elif (view == 'T-Acc'):
            # Filtering The Signals
            main_data['T-AccX'] = helper_functions.butter_lowpass_filter(main_data['T-AccX'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-AccY'] = helper_functions.butter_lowpass_filter(main_data['T-AccY'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-AccZ'] = helper_functions.butter_lowpass_filter(main_data['T-AccZ'], cutoff=3.667, fs=33.3,order=5)

            # Resultant Calculation
            resultant_value = helper_functions.resultant(main_data['T-AccX'], main_data['T-AccY'], main_data['T-AccZ'])
            return {
                'data': [go.Scatter(
                    x=list(range(len(list(resultant_value)))),
                    y=list(resultant_value),
                    mode='lines',
                    line=dict(
                    color='#F0C808',
                    width=2)
                )],
                'layout': go.Layout(
                    xaxis={'title': 'Time Epoch'},
                    yaxis={'title': 'Value'},
                    showlegend=False,
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                    hovermode='closest'
                )
            }
        elif(view=='T-Gyro'):
            # Filtering The Signals
            main_data['T-GyroX'] = helper_functions.butter_lowpass_filter(main_data['T-GyroX'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-GyroY'] = helper_functions.butter_lowpass_filter(main_data['T-GyroY'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-GyroZ'] = helper_functions.butter_lowpass_filter(main_data['T-GyroZ'], cutoff=3.667, fs=33.3,order=5)

            # Resultant Calculation
            resultant_value = helper_functions.resultant(main_data['T-GyroX'], main_data['T-GyroY'], main_data['T-GyroZ'])
            return {
                'data': [go.Scatter(
                    x=list(range(len(list(resultant_value)))),
                    y=list(resultant_value),
                    mode='lines',
                    line=dict(
                    color='#0094C6',
                    width=2)
                )],
                'layout': go.Layout(
                    xaxis={'title': 'Time Epoch'},
                    yaxis={'title': 'Value'},
                    showlegend=False,
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
                    hovermode='closest'
                )
            }

## 2. For Data Table

@app.callback(
    Output('Output-Data-Table','children'),
    [Input('Dataset','value')]
)
def update_table(data):
    if not data:
        return html.Div()

    else:

        data_dir = 'Data/'

        data_list = []

        for i in data:
            dataset_loop = pd.read_csv(data_dir + i + '.csv', sep=',', encoding="utf-8", error_bad_lines=False,low_memory=False)  # Reading Test Datasets
            data_list.append(dataset_loop)  # Data Objects
        main_data = pd.concat(data_list, axis=0, ignore_index=True)

        #Dropping Use Less Columns
        main_data.drop(['iFrame','CNT','SS_Label','Dataset_ID','Breed','Size','Gender','Age','P_ID','Label_Numeric'], axis=1, inplace=True)

        return dash_table.DataTable(
                    id='table',
                    columns=[{"name": i, "id": i} for i in main_data.columns],
                    data=main_data.to_dict("rows"),
                    style_table={'overflowX':'scroll','overflowY':'scroll','maxHeight':'300px','border': 'thin #003F5C solid'},
                    style_cell={'textAlign': 'center', 'width':'150px'},
                    style_header={
                                'backgroundColor': 'rgb(230, 230, 230)',
                                'fontWeight': 'bold'
                    },
                    style_data_conditional=[{
                                'if': {'column_id': 'Label'},
                                'backgroundColor': '#3D9970',
                                'color': 'white'}
                    ]
        )

#3. Testing Distribution
@app.callback(
    Output('dist_area','figure'),
    [Input('Dataset','value')])

def update_dist(data):
    if not data:
        return {
            'data': [],
            'layout': go.Layout(
                xaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                },

                yaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                }
            )
        }

    else:
        data_dir = 'Data/'

        data_list = []

        for i in data:
            dataset_loop = pd.read_csv(data_dir + i + '.csv', sep=',', encoding="utf-8", error_bad_lines=False,low_memory=False)  # Reading Test Datasets
            data_list.append(dataset_loop)  # Data Objects
        main_data = pd.concat(data_list, axis=0, ignore_index=True)

        return {
            'data':[go.Histogram(
                    x=main_data['Label'].values.tolist(),
                    marker=dict(color='#00BFB2')
            )],
            'layout':go.Layout(
                xaxis={'title':'Labels'},
                yaxis={'title':'Number of Samples'},
                margin={'l': 40, 'b': 60, 't': 40, 'r': 10},
                hovermode='closest')

        }

#4. 3D Distribution

@app.callback(
    Output('3d_area','figure'),
    [Input('Dataset','value'),
    Input('view_choice','value')])

def update_graph(data,view):
    if not data:
        return {
            'data': [],
            'layout': go.Layout(
                xaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                },

                yaxis={
                    'showticklabels': False,
                    'ticks': '',
                    'showgrid': False,
                    'zeroline': False
                }
            )
        }

    else:
        data_dir = 'Data/'

        data_list = []

        for i in data:
            dataset_loop = pd.read_csv(data_dir + i + '.csv', sep=',', encoding="utf-8", error_bad_lines=False,low_memory=False)  # Reading Test Datasets
            data_list.append(dataset_loop)  # Data Objects
        main_data = pd.concat(data_list, axis=0, ignore_index=True)

        #Local Lists
        data_segment=[]
        clusters=[]
        colors=['#00BFB2','#0094C6','#71F79F','#2176FF','#9FFFCB','#D99AC5','#F7A072','#044B7F']

        if (view == 'N-Acc'):
            # Filtering The Signals
            main_data['N-AccX'] = helper_functions.butter_lowpass_filter(main_data['N-AccX'], cutoff=3.667, fs=33.3,order=5)
            main_data['N-AccY'] = helper_functions.butter_lowpass_filter(main_data['N-AccY'], cutoff=3.667, fs=33.3,order=5)
            main_data['N-AccZ'] = helper_functions.butter_lowpass_filter(main_data['N-AccZ'], cutoff=3.667, fs=33.3,order=5)

            for i in range(len(main_data['Label'].unique())):
                name=main_data['Label'].unique()[i]
                color=colors[i]
                x = main_data[main_data['Label']==name]['N-AccX']
                y = main_data[main_data['Label']==name]['N-AccY']
                z = main_data[main_data['Label']==name]['N-AccZ']
                trace = dict(
                    name=name,
                    showlegend=False,
                    x=x, y=y, z=z,
                    type="scatter3d",
                    mode='markers',
                    marker=dict(size=3, color=color, line=dict(width=0)))
                data_segment.append(trace)
            layouti = dict(
                width=500,
                height=300,
                margin={'l': 40, 'b': 20, 't': 10, 'r': 10},
                scene=dict(
                    xaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    yaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    zaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7),
                    aspectmode='manual'
                ),
            )
            return dict(data=data_segment, layout=layouti)

        elif (view == 'N-Gyro'):
            # Filtering The Signals
            main_data['N-GyroX'] = helper_functions.butter_lowpass_filter(main_data['N-GyroX'], cutoff=3.667, fs=33.3,order=5)
            main_data['N-GyroY'] = helper_functions.butter_lowpass_filter(main_data['N-GyroY'], cutoff=3.667, fs=33.3,order=5)
            main_data['N-GyroZ'] = helper_functions.butter_lowpass_filter(main_data['N-GyroZ'], cutoff=3.667, fs=33.3,order=5)

            for i in range(len(main_data['Label'].unique())):
                name=main_data['Label'].unique()[i]
                color=colors[i]
                x = main_data[main_data['Label']==name]['N-GyroX']
                y = main_data[main_data['Label']==name]['N-GyroY']
                z = main_data[main_data['Label']==name]['N-GyroZ']
                trace = dict(
                    name=name,
                    showlegend=False,
                    x=x, y=y, z=z,
                    type="scatter3d",
                    mode='markers',
                    marker=dict(size=3, color=color, line=dict(width=0)))
                data_segment.append(trace)
            layouti = dict(
                width=500,
                height=300,
                margin={'l': 40, 'b': 20, 't': 10, 'r': 10},
                scene=dict(
                    xaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    yaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    zaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7),
                    aspectmode='manual'
                ),
            )
            return dict(data=data_segment, layout=layouti)

        elif(view == 'T-Acc'):
            # Filtering The Signals
            main_data['T-AccX'] = helper_functions.butter_lowpass_filter(main_data['T-AccX'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-AccY'] = helper_functions.butter_lowpass_filter(main_data['T-AccY'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-AccZ'] = helper_functions.butter_lowpass_filter(main_data['T-AccZ'], cutoff=3.667, fs=33.3,order=5)

            for i in range(len(main_data['Label'].unique())):
                name=main_data['Label'].unique()[i]
                color=colors[i]
                x = main_data[main_data['Label']==name]['T-AccX']
                y = main_data[main_data['Label']==name]['T-AccY']
                z = main_data[main_data['Label']==name]['T-AccZ']
                trace = dict(
                    name=name,
                    showlegend=False,
                    x=x, y=y, z=z,
                    type="scatter3d",
                    mode='markers',
                    marker=dict(size=3, color=color, line=dict(width=0)))
                data_segment.append(trace)
            layouti = dict(
                width=500,
                height=300,
                margin={'l': 40, 'b': 20, 't': 10, 'r': 10},
                scene=dict(
                    xaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    yaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    zaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7),
                    aspectmode='manual'
                ),
            )
            return dict(data=data_segment, layout=layouti)

        elif (view == 'T-Gyro'):
            # Filtering The Signals
            main_data['T-GyroX'] = helper_functions.butter_lowpass_filter(main_data['T-GyroX'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-GyroY'] = helper_functions.butter_lowpass_filter(main_data['T-GyroY'], cutoff=3.667, fs=33.3,order=5)
            main_data['T-GyroZ'] = helper_functions.butter_lowpass_filter(main_data['T-GyroZ'], cutoff=3.667, fs=33.3,order=5)

            for i in range(len(main_data['Label'].unique())):
                name = main_data['Label'].unique()[i]
                color = colors[i]
                x = main_data[main_data['Label'] == name]['T-GyroX']
                y = main_data[main_data['Label'] == name]['T-GyroY']
                z = main_data[main_data['Label'] == name]['T-GyroZ']
                trace = dict(
                    name=name,
                    showlegend=False,
                    x=x, y=y, z=z,
                    type="scatter3d",
                    mode='markers',
                    marker=dict(size=3, color=color, line=dict(width=0)))
                data_segment.append(trace)
            layouti = dict(
                width=500,
                height=300,
                margin={'l': 40, 'b': 20, 't': 10, 'r': 10},
                scene=dict(
                    xaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    yaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    zaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    aspectratio=dict(x=1, y=1, z=0.7),
                    aspectmode='manual'
                ),
            )
            return dict(data=data_segment, layout=layouti)

#5 Model Output And Analysis
@app.callback(
    Output('eval_section','children'),
    [Input('click_button_1','n_clicks')],
    [State('Dataset','value')])

def Evaluation(n_clicks, data):
    if n_clicks!=None:
        if not data:
            return \
            html.Div(className='row',children=[
                html.Br(),
                html.Br(),
                html.Br(),

                html.Div(className='col-lg-4 note-tag-1', children=[
                    dcc.Markdown(children='''
                    >**Note :** Please Select Some Data To Initialize The Model.''')
                ], style={'margin-top': '12px','display':'inline-block'}),

            ], style={'text-align':'center'})

        else:
            data_dir = 'Data/'

            pred_data_list = []

            for i in data:
                dataset_loop = pd.read_csv(data_dir + i + '.csv', sep=',', encoding="utf-8", error_bad_lines=False,low_memory=False)  # Reading Test Datasets
                print(dataset_loop.columns)
                #Right Now we have the Array of out Activity Data
                #Calling The Model
                predicted_data=Model.model_call(dataset_loop)

                pred_data_list.append(predicted_data)# Data Objects

            main_data=pd.concat(pred_data_list, axis=0, ignore_index=True)

            ##################################################################### First Output Segment ################################################################################

            predict_save_columns = ['Sample', 'N-AccX', 'N-AccY', 'N-AccZ', 'N-GyroX', 'N-GyroY', 'N-GyroZ', 'T-AccX', 'T-AccY',
                                    'T-AccZ', 'T-GyroX', 'T-GyroY', 'T-GyroZ','Breed', 'Size', 'Gender', 'Age', 'Information', 'Label', 'Predictions']

            file_name=str(main_data['Label'].unique())
            #Saving The Output File
            main_data[predict_save_columns].to_csv('Output/Prediction_'+str(n_clicks)+'_'+file_name+'.csv',index=False)

            ####################################################### Generating Table For View ### Second Output Segment ################################################################
            #Also For UI - Output Data
            output_view_data=helper_functions.calc_accuracy_per_sample(main_data)
            output_view_data.to_csv('Output/Prediction_Accuracy_'+str(n_clicks)+file_name+'.csv',index=False)

            #Overall Performance

            overall_performance_data=helper_functions.calc_accuracy_per_activity(main_data)




            return \
                html.Div(className='container',children=[
                    html.Div(className='row',children=[
                        html.Div(className='col-lg-6 white-bg',children=[
                            html.H4("Output Data", className='selector-sub'),
                            dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in output_view_data.columns],
                                data=output_view_data.to_dict("rows"),
                                style_table={'overflowX': 'scroll', 'overflowY': 'scroll', 'maxHeight': '300px',
                                             'border': 'thin #003F5C solid'},
                                style_cell={'textAlign': 'center', 'width': '190px'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                },
                                style_data_conditional=[{
                                    'if': {'column_id': 'Label'},
                                    'backgroundColor': '#3D9970',
                                    'color': 'white'},
                                    {
                                        'if': {'column_id': 'Predictions'},
                                        'backgroundColor': '#D55672',
                                        'color': 'white'
                                    }
                                ]
                            )

                        ]),
                        html.Div(className='col-lg-6 white-bg', children=[
                            html.H4("Overall Performance", className='selector-sub'),
                            dash_table.DataTable(
                                id='table',
                                columns=[{"name": i, "id": i} for i in overall_performance_data.columns],
                                data=overall_performance_data.to_dict("rows"),
                                style_table={'overflowX': 'scroll', 'overflowY': 'scroll', 'maxHeight': '300px'},
                                style_cell={'textAlign': 'center', 'width': '250px'},
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                }
                            )


                        ])
                    ])

            ])










if __name__ == '__main__':
    app.run_server(debug=True, port=8050)