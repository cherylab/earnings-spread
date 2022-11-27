import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from plotly.graph_objs.scatter.marker import Line
from plotly.subplots import make_subplots
import numpy as np
import plot_settings


# do date filter and recalculate stds
def calc_stds(d,
              start_date,
              end_date,
              y_col):
    df_period = d.query(f"date>=@start_date & date<=@end_date").reset_index(drop=True)
    mean = df_period[y_col].mean()
    std = df_period[y_col].std()
    std_1up = mean + std
    std_1down = mean - std
    std_2up = mean + 2*std
    std_2down = mean - 2*std

    return df_period, mean, std_1up, std_1down, std_2up, std_2down


# create plot with option for mean/std lines or normal grid
def std_plot_with_button(plot_df,
                         x_col, # df column along x axis
                         y_col, # df column along y axis
                         y_label, # y axis label
                         title, # plot title
                         mean, # mean value for date range
                         up1, # +1std value for date range
                         down1, # -1std value for date range
                         up2, # +2std value for date range
                         down2, # -2std value for date range
                         checkbox_key, # unique key for checkbox input
                         more_distance=False
                         ):
    st.write("<br><br>", unsafe_allow_html=True)
    showstd = st.checkbox('Show mean and \u03C3 lines', value=True, key=checkbox_key)
    st.write("<br>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=plot_df[x_col],
                             y=plot_df[y_col]))

    fig.update_traces(marker=dict(size=3))

    if showstd:
        hline_color = "#191919"  # "#848484"

        fig.add_hline(y=mean,
                      line_width=2,
                      line_color=hline_color,
                      )

        fig.add_hline(y=up1,
                      line_dash="dash",
                      line_width=1,
                      line_color=hline_color,
                      )

        fig.add_hline(y=up2,
                      line_dash="dot",  # "dash"
                      line_width=1,
                      line_color=hline_color,
                      )

        fig.add_hline(y=down1,
                      line_dash="dash",
                      line_width=1,
                      line_color=hline_color,
                      )

        fig.add_hline(y=down2,
                      line_dash="dot",  # "dash"
                      line_width=1,
                      line_color=hline_color,
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

        fig.update_layout(annotations=annotations,
                          template=plot_settings.dockstreet_template,
                          height=500)

    else:
        fig.update_layout(template=plot_settings.dockstreet_template)

    fig.update_xaxes(showgrid=False,
                     range=[plot_df.date.min(), plot_df.date.max()])

    fig.update_yaxes(showgrid=False if showstd else True,
                     zeroline=False,
                     title=y_label,
                     ticksuffix="  ",
                     range=[np.floor(plot_df[y_col].min()) - 1.5,
                            np.ceil(plot_df[y_col].max()) + .05])

    fig.update_layout(font_family="Avenir",
                      font_color="#4c4c4c",
                      font_size=14,
                      showlegend=False,
                      margin=dict(t=70),
                      title=dict(font_size=22,
                                 text=f"<b>{title}</b>",
                                 xref='container',
                                 yref='container',
                                 x=.02,
                                 y=1.0
                                 ),
                      plot_bgcolor='white'
                      #                   title_pad_b=1000
                      )

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
                       ay=plot_df[y_col].max()-.75 if more_distance else plot_df[y_col].max()-.25,
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
                       ay=np.floor(plot_df[y_col].min()) - .6, #if showstd else .3,
                       arrowhead=1,
                       arrowsize=1.,
                       arrowside='start',
                       arrowwidth=1.5,
                       arrowcolor="#767676",
                       showarrow=True,
                       font=dict(size=15,
                                 color="#767676"
                                 ))

    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                y=1.,
                x=.01,
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
                range=[plot_df[x_col].min(), plot_df[x_col].max()]
            ),
            type="date"
        )
    )

    for ser in fig['data']:
        ser['hovertemplate'] = "%{x|%b %-d, %Y}, %{y:.2f}<extra></extra>"

    return fig

# plot oscillator plot for 1 column of numbers
def oscillator_plot(plot_df_pre, #df with the data
                    col_use, #df column oscillator based on
                    col_name, #checklist name for the col_use column
                    num_wks, #number of weeks to base calcs on
                    start_date, #start date in chosen date inputs
                    end_date, #end date in chosen date inputs
                    line_color #color to use for line plot
                    ):

    plot_df = plot_df_pre.copy()

    plot_df = plot_df.assign(high=lambda t: t[col_use].rolling(num_wks).max(),
                       low=lambda t: t[col_use].rolling(num_wks).min(),
                       kperc=lambda t: (t[col_use] - t.low) / (t.high - t.low))

    plot_df = plot_df[(plot_df.date >= start_date) & (plot_df.date <= end_date)]

    # second reference plot
    oscfig = go.Figure()
    oscfig.add_trace(go.Scatter(x=plot_df['date'],
                                y=plot_df['kperc'],
                                line=dict(width=2.25, color=line_color),
                                name="Oscillator"))
    oscfig.update_layout(template=plot_settings.dockstreet_template,
                         plot_bgcolor='white',
                         hovermode='x',
                         font_family="Avenir",
                         font_color="#4c4c4c",
                         showlegend=False,
                         height=300,
                         margin=dict(b=25, t=25)
                         )
    oscfig.update_yaxes(title_text=f"<b>{col_name.upper()}</b><br>{num_wks} Wk Oscillator",
                        tickformat=",%",
                        titlefont_size=16,
                        zeroline=False,
                        color="#4c4c4c",
                        tickcolor="#4c4c4c",
                        tickfont_color="#4c4c4c",
                        tickfont_size=14,
                        showgrid=True,
                        ticksuffix="  ",
                        title_standoff=10,
                        range=[-.05, plot_df['kperc'].max() + .025])

    oscfig.update_xaxes(tickcolor="#4c4c4c",
                        tickfont_color="#4c4c4c",
                        tickfont_size=14,
                        showgrid=False, )

    oscfig.add_hline(y=.2,
                     line_dash="dot",
                     line_width=1.5,
                     line_color="#191919",
                     )

    oscfig.add_hline(y=.8,
                     line_dash="dot",
                     line_width=1.5,
                     line_color="#191919",
                     )

    oscfig.add_hrect(y0=.2,
                     y1=.8,
                     line_width=0,
                     fillcolor="#d7d7d7",
                     opacity=0.2
                     )

    for ser in oscfig['data']:
        ser['hovertemplate'] = "%{x|%b %-d, %Y}<br>%{y:,.0%}<extra></extra>"
        
    return oscfig