import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from plotly.graph_objs.scatter.marker import Line
from plotly.subplots import make_subplots
import numpy as np
import plot_settings
# import requests
# import io
from flask_caching import Cache
import logins

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
auth = dash_auth.BasicAuth(app,logins.USERNAME_PASSWORD_PAIRS)
server = app.server

cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

# tab styling since tabs can't be styled from .css file
tabs_style = {
    'height': '44px',
    'marginTop': '10px',
    'marginBottom': '10px',
    'width': '100%',
    'paddingLeft': '10%',
    'paddingRight': '10%',
}
tab_style = {
    'borderBottom': '1px solid  #6f6f6f',
    'borderTop': '0px',
    'borderRight': '0px',
    'borderLeft': '0px',
    'padding': '10px',
    'fontWeight': 'normal',
    'backgroundColor':  '#eaeaea',
    'color': '#4c4c4c',
    'fontWeight': '500'
}
tab_selected_style = {
    'backgroundColor': '#eaeaea',
    'color': '#4c4c4c',
    'padding': '10px',
    'borderBottom': '1px solid  #6f6f6f',
    'borderTop': '1px solid  #6f6f6f',
    'borderRight': '0px solid  #6f6f6f',
    'borderLeft': '0px solid  #6f6f6f',
    'fontWeight': '900',
}

# function to get file from google drive
# https://drive.google.com/file/d/16NBIP4qGtBkNbcxfUElMDjWiADryMa-G/view?usp=sharing
TIMEOUT = 60
@cache.memoize(timeout=TIMEOUT)
def pull_google_drive(url):
    file_id = url.split('/')[-2]
    dwn_url = "https://drive.google.com/uc?id=" + file_id
    tmp = pd.read_excel(dwn_url)
    return tmp

# filter to columns needed and format names
def reformat_df(d):
    tmp = d.filter(['Date', 'Spread', 'SPX_Price'])
    tmp.columns = [x.lower() for x in tmp.columns]
    tmp = tmp.assign(date=lambda t: pd.to_datetime(t.date),
                   inverse=lambda t: 1 / (t.spread / 100))
    tmp = tmp.sort_values(by='date').reset_index(drop=True)

    return tmp

# do date filter and recalculate stds
def calc_stds(d, start_date, end_date):
    df_period = d.query(f"date>=@start_date & date<=@end_date").reset_index(drop=True)
    mean = df_period.spread.mean()
    std = df_period.spread.std()
    std_1up = mean + std
    std_1down = mean - std
    std_2up = mean + 2*std
    std_2down = mean - 2*std

    return df_period, mean, std_1up, std_1down, std_2up, std_2down

def create_graph(plot_df, mean=0, up1=0, down1=0, up2=0, down2=0, linevalue='meanstd'):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plot_df['date'],
                             y=plot_df['spread']))

    fig.update_traces(marker=dict(size=3))

    if linevalue=='meanstd':
        hline_color = "black"  # "#848484"

        fig.add_shape(type='line',
                      x0=plot_df['date'].min(),
                      x1=plot_df['date'].max(),
                      y0=mean,
                      y1=mean,
                      xref='x',
                      yref='y',
                      line=dict(color=hline_color,
                                width=1),
        )

        fig.add_shape(type='line',
                      x0=plot_df['date'].min(),
                      x1=plot_df['date'].max(),
                      y0=up1,
                      y1=up1,
                      xref='x',
                      yref='y',
                      line=dict(color=hline_color,
                                width=1,
                                dash='dash'),
                      )

        fig.add_shape(type='line',
                      x0=plot_df['date'].min(),
                      x1=plot_df['date'].max(),
                      y0=up2,
                      y1=up2,
                      xref='x',
                      yref='y',
                      line=dict(color=hline_color,
                                width=1,
                                dash='dot'),
                      )

        fig.add_shape(type='line',
                      x0=plot_df['date'].min(),
                      x1=plot_df['date'].max(),
                      y0=down1,
                      y1=down1,
                      xref='x',
                      yref='y',
                      line=dict(color=hline_color,
                                width=1,
                                dash='dash'),
                      )

        fig.add_shape(type='line',
                      x0=plot_df['date'].min(),
                      x1=plot_df['date'].max(),
                      y0=down2,
                      y1=down2,
                      xref='x',
                      yref='y',
                      line=dict(color=hline_color,
                                width=1,
                                dash='dot'),
                      )

        # annotations
        annotations = []

        points = [up2, up1, mean, down1, down2]
        labels = ["+2\u03C3", "+1\u03C3", "mean", "-1\u03C3", "-2\u03C3"]

        for p, l in zip(points, labels):
            annotations.append(dict(xref='paper',
                                    x=1.005,
                                    y=p,
                                    xanchor='left',
                                    yanchor='middle',
                                    align='left',
                                    text=f"{p:.2f} ({l})",
                                    showarrow=False,
                                    font=dict(size=12, color=hline_color)
                                    ))

        fig.update_layout(annotations=annotations)

    fig.add_annotation(x=0.03,
                       y=mean + .2,
                       xref='paper',
                       yref='y',
                       xanchor='left',
                       align='left',
                       borderpad=5,
                       text="Cheaper",
                       axref='pixel',
                       ayref='y',
                       ax=0.25,
                       ay=mean + 3.8,
                       arrowhead=1,
                       arrowsize=1.,
                       arrowside='start',
                       arrowwidth=1.5,
                       arrowcolor="#767676",
                       showarrow=True,
                       font=dict(size=15,
                                 color="#767676"
                                 ))

    fig.add_annotation(x=0.03,
                       y=mean - .2,
                       xref='paper',
                       yref='y',
                       xanchor='left',
                       align='left',
                       borderpad=5,
                       text="More Expensive",
                       axref='pixel',
                       ayref='y',
                       ax=0.25,
                       ay=mean - 3.8,
                       arrowhead=1,
                       arrowsize=1.,
                       arrowside='start',
                       arrowwidth=1.5,
                       arrowcolor="#767676",
                       showarrow=True,
                       font=dict(size=15,
                                 color="#767676"
                                 ))



    fig.update_xaxes(showgrid=False if linevalue=='meanstd' else True,
                     range=[plot_df.date.min(), plot_df.date.max()])

    fig.update_yaxes(showgrid=False if linevalue=='meanstd' else True,
                     zeroline=False,
                     title='Equity Risk Premium',
                     ticksuffix="  ",
                     range=[-.5, np.ceil(plot_df.spread.max())])

    fig.update_layout(font_family="Avenir",
                      font_color="#4c4c4c",
                      font_size=14,
                      showlegend=False,
                      template=plot_settings.dockstreet_template,
                      margin=dict(t=10, b=10)
                      )

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(count=2,
                         label="2y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True,
                range=[plot_df.date.min(), plot_df.date.max()]
            ),
            type="date"
        )
    )

    for ser in fig['data']:
        ser['hovertemplate'] = "%{x|%b %-d, %Y}, %{y:.2f}<extra></extra>"

    return fig

def create_SPvsPE_graph(plot_df):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df.date, y=df.inverse, name="Bond Adj. P/E Ratio"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.date, y=df.spx_price, name="S&P 500 Price", line=dict(width=1.5)),
        secondary_y=True,
    )

    fig.update_layout(template=plot_settings.dockstreet_template,
                      margin=dict(t=10),
                      hovermode='x',
                      font_family="Avenir",
                      font_color="#4c4c4c",
                      # title=dict(font_size=22,
                      #            text="<b>Spread between S&P price and P/E ratio is widening</b>",
                      #            x=0.04,
                      #            y=0.93
                      #            ),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.1,
                          xanchor="left",
                          x=0,
                          font=dict(size=14)
                      ))

    fig.update_yaxes(title_text="<br>S&P Price",
                     titlefont_size=18,
                     color="#767676",
                     tickcolor="#767676",
                     tickfont_color="#767676",
                     tickfont_size=13,
                     showgrid=False,
                     tickformat="$,",
                     tickprefix="  ",
                     title_standoff=5,
                     secondary_y=True)

    fig.update_yaxes(title_text="Bond Adjusted P/E<br>",
                     titlefont_size=18,
                     title_standoff=20,
                     color=plot_settings.color_list[0],
                     tickcolor=plot_settings.color_list[0],
                     tickfont_color=plot_settings.color_list[0],
                     tickfont_size=13,
                     ticksuffix="   ",
                     range=[df.inverse.min() - (df.inverse.min() % 10) - 2, df.inverse.max() + (df.inverse.max() % 10)],
                     secondary_y=False)

    fig.update_xaxes(showgrid=False)

    fig.add_annotation(x=df.date.max(),
                       y=df.spx_price.values[-1],
                       xref='x',
                       yref='y2',
                       xanchor='left',
                       align='left',
                       borderpad=5,
                       text=f"${df.spx_price.values[-1]:,.0f}",
                       showarrow=False,
                       font=dict(size=12,
                                 color="#767676")
                       )

    fig.add_annotation(x=df.date.max(),
                       y=df.inverse.values[-1],
                       xref='x',
                       yref='y',
                       xanchor='left',
                       align='left',
                       borderpad=5,
                       text=f"{df.inverse.values[-1]:,.1f}",
                       showarrow=False,
                       font=dict(size=12,
                                 color=plot_settings.color_list[0])
                       )

    for ser in fig['data']:
        if ser['name'] == "Bond Adj. P/E Ratio":
            ser['hovertemplate'] = "%{x|%b %-d, %Y}, %{y:,.2f}<extra></extra>"
        else:
            ser['hovertemplate'] = "%{x|%b %-d, %Y}, %{y:$,.2f}<extra></extra>"

    return fig

# load the data from google drive
url = "https://drive.google.com/file/d/16NBIP4qGtBkNbcxfUElMDjWiADryMa-G/view?usp=sharing"
df = pull_google_drive(url)

# format columns
df = reformat_df(df)

df_date_filter, m, u1, d1, u2, d2 = calc_stds(df, df.date.min(), df.date.max())

# first tab graph
updated_figure = create_graph(df_date_filter, mean=m, up1=u1, down1=d1, up2=u2, down2=d2, linevalue='meanstd')

# second tab graph
sp_pe_figure = create_SPvsPE_graph(df)

app.layout = \
    html.Div([
            # header banner with titles
            html.Div(
                children=[
                    html.H1('DOCK STREET ASSET MANAGEMENT',
                            className="header-company-name"),
                    html.P('HISTORICAL EARNINGS SPREAD ANALYSIS', #'Historical earnings spread analysis',
                           className='header-page-name')
                ],
                className="header-banner"
        ),
        # dbc.Container([
            dcc.Tabs([
                dcc.Tab(label='Equity Risk', children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Row([
                                dbc.Col([
                                    # user inputs and submit button
                                    html.Div([
                                        html.Div([
                                            html.Div(children='User inputs',
                                                     className='block-title'),
                                            html.Div(children='', className='title-line'),
                                            html.Div(children='Enter a start and end date:',
                                                     className='input-title'),
                                            dcc.DatePickerRange(id='my_date_range',
                                                                min_date_allowed=df.date.min(),
                                                                max_date_allowed=df.date.max(),
                                                                start_date=df.date.min(),
                                                                end_date=df.date.max()
                                                                ),
                                        ],
                                        ),
                                        html.Div(
                                            children=[
                                                html.Button(id='submit_button',
                                                            n_clicks=0,
                                                            children='Submit Inputs',
                                                            className='button')
                                            ],
                                        )
                                    ],
                                        className='utilities-block'
                                    )
                                ], width=True)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    # checkbox
                                    html.Div(
                                        children=[
                                            html.Div(children='Plot Adjustments',
                                                     className='block-title'),
                                            html.Div(children='', className='title-line'),
                                            dbc.RadioItems(id='plot_lines',
                                                           options=[
                                                               {'label': ' Show mean and \u03C3 lines',
                                                                'value': 'meanstd'},
                                                               {'label': ' Show standard grid', 'value': 'grid'}
                                                           ],
                                                           value='meanstd',
                                                           labelStyle={'display': 'block'},
                                                           inputClassName='radio-input',
                                                           labelClassName='radio-label')
                                        ],
                                        # style={'marginTop': '200px'},
                                        className='utilities-block'
                                    )
                                ], width=True)
                            ])
                        ], width=4),
                        dbc.Col([
                            # graph
                            html.Div(
                                children=[
                                    html.Div(
                                        children='Equity risk premium mainly between 15yr mean and -1\u03C3 in recent months',
                                        className='block-title'),
                                    html.Div(children='', className='title-line'),
                                    dcc.Graph(id='my_graph',
                                              figure=updated_figure,
                                              style={'height': '83%'},
                                              className='content-block')
                                ],
                                # style={'display': 'inline-block'},
                                # style={'width': '72%', 'display': 'inline-block', 'verticalAlign':'top', 'height':'450px'},
                                className='utilities-block'
                            )
                        ], width=8)
                    ], justify='around')
                ], style=tab_style, selected_style=tab_selected_style),


                dcc.Tab(label='S&P vs P/E Ratio', children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div(
                                children=[
                                    html.Div(children='Spread between S&P price and P/E ratio is widening',
                                             className='block-title'),
                                    html.Div(children='', className='title-line'),
                                    dcc.Graph(id='my_sp_pe_graph',
                                              figure=sp_pe_figure,
                                              style={'height': '94%', 'paddingLeft': '12px'},
                                              className='content-block')
                                ],
                                className='utilities-block'
                            )
                        ])
                    ])
                ], style=tab_style, selected_style=tab_selected_style)
            ], style=tabs_style)
        # ])
    ])

@app.callback(Output('my_graph','figure'),
              [Input('submit_button','n_clicks')],
              [
                  State('my_date_range','start_date'),
                  State('my_date_range','end_date'),
                  State('plot_lines', 'value'),
              ])
def callback_dates(n_clicks, start_date, end_date, linevalue):
    # when it gets passed into the input, converts it to a string
    start = datetime.strptime(start_date[:10], '%Y-%m-%d')
    end = datetime.strptime(end_date[:10], '%Y-%m-%d')

    df_date_filter, m, u1, d1, u2, d2 = calc_stds(df, start, end)

    updated_figure = create_graph(df_date_filter, mean=m, up1=u1, down1=d1, up2=u2, down2=d2, linevalue=linevalue)

    return updated_figure

# @app.callback(Output('my_graph','figure'),
#     [Input('plot_lines','value'),
#      Input('my_date_range','start_date'),
#      Input('my_date_range','end_date')]
# )
# def callback_linevalue(linevalue, start_date, end_date):
#     start = datetime.strptime(start_date[:10], '%Y-%m-%d')
#     end = datetime.strptime(end_date[:10], '%Y-%m-%d')
#
#     df_date_filter, m, u1, d1, u2, d2 = calc_stds(df, start, end)
#
#     updated_figure = create_graph(df_date_filter, mean=m, up1=u1, down1=d1, up2=u2, down2=d2, linevalue=linevalue)
#
#     return updated_figure

if __name__ == '__main__':
    app.run_server(debug=True)