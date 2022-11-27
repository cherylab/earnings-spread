import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import *
from plotly.graph_objs.scatter.marker import Line
from plotly.subplots import make_subplots
import numpy as np
import plot_settings
from multiapp import MultiApp
import plot_functions
# import httpagentparser
import requests
from streamlit.report_thread import get_report_ctx

# ctx = get_report_ctx()
# print(ctx)

# headers = {'User-Agent':'something'}
# s = requests.get('http://google.com', headers=headers)
# print(headers['User-Agent'])
# print(httpagentparser.detect(s))

st.set_page_config(layout='wide')

# PULL FIRST SPREADSHEET
# function to get file from google drive
@st.cache
def pull_google_drive(url):
    file_id = url.split('/')[-2]
    dwn_url = "https://drive.google.com/uc?id=" + file_id

    tmp = pd.read_excel(dwn_url)

    return tmp

# filter to columns needed and format names
def reformat_df(d):
    tmp = d.filter(['Date', 'Spread', 'SPX_Price', '10YearYield', 'NTM-EarningsYield'])
    tmp.columns = [x.replace('-','_').lower() for x in tmp.columns]
    tmp = tmp.assign(date=lambda t: pd.to_datetime(t.date),
                     inverse=lambda t: 1 / (t.spread / 100),
                     earnings_div_bond=lambda t: t.ntm_earningsyield / t['10yearyield'],
                     diff_div_bond=lambda t: (t.ntm_earningsyield - t['10yearyield']) / t['10yearyield'])
    tmp = tmp.sort_values(by='date').reset_index(drop=True)
    return tmp

# load the data from google drive
url = "https://drive.google.com/file/d/16NBIP4qGtBkNbcxfUElMDjWiADryMa-G/view?usp=sharing"
df_pre = pull_google_drive(url)

# format columns
df = reformat_df(df_pre)
print('max date', df.date.max())

# PULL CTO SPREADSHEET
# function to get cto file from google drive
@st.cache
def pull_google_drive_cot(url, sheetname='spx'):
    file_id = url.split('/')[-2]
    dwn_url = "https://drive.google.com/uc?id=" + file_id

    tmp = pd.read_excel(dwn_url, sheet_name=sheetname) #, usecols=['Date','Leveraged Funds','Asset','Dealer','spx'])

    return tmp

# reformat columns in cto df
def cftcdf_reformat(d):
    tmp = d.copy()
    tmp.columns = [x.lower().replace(' ', '_') for x in tmp.columns]
    tmp['date'] = pd.to_datetime(tmp['date'])

    # num_cols = ['leveraged_funds', 'asset', 'dealer']
    # for c in num_cols:
    #     tmp[c] = tmp[c].str.replace(',', '')
    #     tmp[c] = pd.to_numeric(tmp[c])

    tmp = tmp.assign(lev_deal=lambda t: t.leveraged_funds + t.dealer)
    # tmp = tmp.sort_values(by='date')

    return tmp

# load the cftc data
url_cftc = "https://docs.google.com/spreadsheets/d/1a1FXFASFqpab-EJEA5Ige6dp5GCiKIL5/edit?usp=sharing&ouid=109079795382383182623&rtpof=true&sd=true"

# run the cto spreadsheet pull and reformat in one go
def pull_cot_data(url_cftc, sheet):
    cftcdf = pull_google_drive_cot(url_cftc, sheetname=sheet)

    # format oftc df
    cftc = cftcdf_reformat(cftcdf)
    cftc = cftc.sort_values(by='date', ascending=False)
    return cftc

# -----------------------------------------------------------------------
# inverse graph function
def create_inverse_graph(df_all):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df_all.date, y=df_all.inverse,
                   name="Bond Adj. P/E Ratio"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df_all.date,
                   y=df_all.spx_price,
                   name="S&P 500 Price",
                   line=dict(width=1.5)),
        secondary_y=True,
    )

    fig.update_layout(template=plot_settings.dockstreet_template,
                      plot_bgcolor='white',
                      hovermode='x',
                      font_family="Avenir",
                      font_color="#4c4c4c",
                      margin=dict(t=75),
                      title=dict(font_size=22,
                                 text="<b>S&P price and interest rate adjusted P/E ratio</b>",
                                 x=0.015,
                                 y=1.,
                                 xref='container',
                                 yref='container'
                                 ),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.0,
                          xanchor="left",
                          x=.0,
                          font=dict(size=14)
                      ))

    fig.update_yaxes(title_text="S&P Price",
                     titlefont_size=17,
                     color=plot_settings.color_list[1],
                     tickcolor=plot_settings.color_list[1],
                     tickfont_color=plot_settings.color_list[1],
                     tickfont_size=13,
                     showgrid=False,
                     tickformat="$,",
                     tickprefix="  ",
                     title_standoff=20,
                     secondary_y=True)

    fig.update_yaxes(title_text="Bond Adjusted P/E",
                     titlefont_size=17,
                     color=plot_settings.color_list[0],
                     tickcolor=plot_settings.color_list[0],
                     tickfont_color=plot_settings.color_list[0],
                     tickfont_size=13,
                     ticksuffix="   ",
                     range=[df_all.inverse.min() - (df_all.inverse.min() % 10) - 2,
                            df_all.inverse.max() + (df_all.inverse.max() % 10)],
                     secondary_y=False)

    fig.update_xaxes(showgrid=False)

    fig.add_annotation(x=df_all.date.max(),
                       y=df_all.spx_price.values[-1],
                       xref='x',
                       yref='y2',
                       xanchor='left',
                       align='left',
                       borderpad=5,
                       text=f"${df_all.spx_price.values[-1]:,.0f}",
                       showarrow=False,
                       font=dict(size=12,
                                 color=plot_settings.color_list[1])
                       )

    fig.add_annotation(x=df_all.date.max(),
                       y=df_all.inverse.values[-1],
                       xref='x',
                       yref='y',
                       xanchor='left',
                       align='left',
                       borderpad=5,
                       text=f"{df_all.inverse.values[-1]:,.0f}",
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

# ---------------------------------------------------------------------
# FIRST PAGE
def earnings_recalc():
    st.title('Earnings Spread Analysis')

    # st.sidebar.write('<br><b>Date Inputs</b>', unsafe_allow_html=True)

    st.sidebar.write('<br><br>', unsafe_allow_html=True)
    with st.sidebar.form(key='date_form'):
        st.write('<b>Date Inputs</b>', unsafe_allow_html=True)
        start_date = st.date_input('Choose a start date',
                                     value=df.date.min(),
                                     min_value=df.date.min(),
                                     max_value=df.date.max(),
                                     key='start')
        end_date = st.date_input('Choose an end date',
                                   value=df.date.max(),
                                   min_value=df.date.min(),
                                   max_value=df.date.max(),
                                   key='end')
        submit_button = st.form_submit_button('Submit', help='Press to recalculate')


    df_date_filter, m, u1, d1, u2, d2 = plot_functions.calc_stds(df, start_date, end_date, 'spread')

    updated_figure = plot_functions.std_plot_with_button(df_date_filter,
                                                        'date',
                                                        'spread',
                                                        'Equity Risk Premium',
                                                        'Equity risk premium mainly between 15yr mean and -1\u03C3 lately',
                                                        m,
                                                        u1,
                                                        d1,
                                                        u2,
                                                        d2,
                                                         'original')

    st.plotly_chart(updated_figure, use_container_width=True)

    st.write(f"The above plot shows the daily S&P 500 Forward Earnings Yield minus the 10 Year US Treasury Bond Yield for {start_date.strftime('%b %-d, %Y')} through {end_date.strftime('%b %-d, %Y')}. "
             f"The forward earnings yield is the inverse of the S&P 500's forward P/E ratio.")

    st.markdown("""
        <style>
        .small-font {
            font-size:12px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="small-font"><br><br>Data from <a href="https://www.factset.com/">FactSet</a></p>',
                unsafe_allow_html=True)


# SECOND PAGE
def something_else():
    st.title('S&P Price vs P/E Ratio')

    pe_figure = create_inverse_graph(df)

    st.write('<br>', unsafe_allow_html=True)

    st.plotly_chart(pe_figure, use_container_width=True)

    st.write("The bond adjusted P/E ratio is calculated by taking the S&P 500 Forward Earnings Yield, subtracting the 10 Year Treasury Bond Yield, "
             "and then taking the inverse of this result to make a P/E ratio. The S&P price is the daily closing price for the index.")

    st.markdown("""
    <style>
    .small-font {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="small-font"><br><br>Data from <a href="https://www.factset.com/">FactSet</a></p>', unsafe_allow_html=True)


# THIRD PAGE
def earnings_divided_yield():
    st.title('Stock to Bond Ratio')

    # sidebar date filter
    st.sidebar.write('<br><br>', unsafe_allow_html=True)
    with st.sidebar.form(key='date_form_third'):
        st.write('<b>Date Inputs</b>', unsafe_allow_html=True)
        start_date_third = st.date_input('Choose a start date',
                                   value=df.date.min(),
                                   min_value=df.date.min(),
                                   max_value=df.date.max(),
                                   key='start_third')
        end_date_third = st.date_input('Choose an end date',
                                 value=df.date.max(),
                                 min_value=df.date.min(),
                                 max_value=df.date.max(),
                                 key='end_third')
        submit_button_third = st.form_submit_button('Submit', help='Press to recalculate')

    # std graph
    df_date_filter_third, m_3, u1_3, d1_3, u2_3, d2_3 = plot_functions.calc_stds(df,
                                                                                 start_date_third,
                                                                                 end_date_third,
                                                                                 'earnings_div_bond')

    updated_figure_third = plot_functions.std_plot_with_button(df_date_filter_third,
                                                         'date',
                                                         'earnings_div_bond',
                                                         'Forward Earnings Yield / Bond Yield',
                                                         'Stock to Bond Ratio',
                                                         m_3,
                                                         u1_3,
                                                         d1_3,
                                                         u2_3,
                                                         d2_3,
                                                        'divide',
                                                         more_distance=True)

    updated_figure_third.update_layout(margin=dict(b=50))

    if (datetime(2020,3,9).date()<=end_date_third) & (datetime(2020,3,9).date()>=start_date_third):
        march_high = df_date_filter_third[df_date_filter_third.date==datetime(2020,3,9)]['earnings_div_bond'].values[0]
        march_stds = f"+{(march_high - m_3) / (u1_3 - m_3):.2f}\u03C3"

        updated_figure_third.add_annotation(x=datetime(2020,3,9).date(),
                                           y=march_high,
                                           xref='x',
                                           yref='y',
                                           xanchor='left',
                                           align='left',
                                           borderpad=5,
                                           text=march_stds,
                                           showarrow=False,
                                           font=dict(size=12,
                                                     color="#4c4c4c"
                                                     ))

    st.plotly_chart(updated_figure_third, use_container_width=True)

    st.write(
        f"The above plot shows the S&P 500 Forward Earnings Yield divided by 10 Year US Treasury Bond Yield for "
        f"{start_date_third.strftime('%b %-d, %Y')} through {end_date_third.strftime('%b %-d, %Y')}. "
        f"The forward earnings yield is the inverse of the S&P 500's forward P/E ratio."
        f"<br><br><br>", unsafe_allow_html=True)

    # ------------------------------------
    # reference plot
    ref_expander = st.beta_expander('Reference Plots', expanded=True)
    with ref_expander:
        st.write('<br>', unsafe_allow_html=True)
        yscale = st.checkbox('Show logarithmic scale on price reference plot', value=False, key='logscale')
        st.write('<br>', unsafe_allow_html=True)

        reffig_price = go.Figure()
        reffig_price.add_trace(go.Scatter(x=df_date_filter_third['date'],
                                          y=df_date_filter_third['spx_price'],
                                          line=dict(width=2),
                                          name="S&P 500 Price"))
        reffig_price.update_layout(template=plot_settings.dockstreet_template,
                                   plot_bgcolor='white',
                                   hovermode='x',
                                   font_family="Avenir",
                                   font_color="#4c4c4c",
                                   title=dict(font_size=22,
                                              text="<b>S&P 500 price</b>",
                                              x=0.015,
                                              y=.99,
                                              xref='container',
                                              yref='container',
                                              ),
                                   margin=dict(t=70)
                                   )
        reffig_price.update_yaxes(title_text="S&P Price",
                                  titlefont_size=16,
                         color="#4c4c4c",
                         tickcolor="#4c4c4c",
                         tickfont_color="#4c4c4c",
                         tickfont_size=14,
                         showgrid=True,
                         tickformat="$,.0f",
                         ticksuffix="  ",
                         title_standoff=20)

        reffig_price.update_xaxes(tickcolor="#4c4c4c",
                         tickfont_color="#4c4c4c",
                         tickfont_size=14,
                         showgrid=False,)

        if yscale:
            reffig_price.update_yaxes(type="log")
            reffig_price.update_layout(height=560)
        else:
            reffig_price.update_layout(height=460)

        for ser in reffig_price['data']:
            ser['hovertemplate'] = "%{x|%b %-d, %Y}<br>%{y:$,.2f}<extra></extra>"


        # second reference plot
        reffig_yield = go.Figure()
        reffig_yield.add_trace(go.Scatter(x=df_date_filter_third['date'],
                                          y=df_date_filter_third['10yearyield'],
                                          line=dict(width=2),
                                          name="10Yr Bond Yield"))
        reffig_yield.add_trace(go.Scatter(x=df_date_filter_third['date'],
                                          y=df_date_filter_third['ntm_earningsyield'],
                                          line=dict(width=2),
                                          name="NTM Earnings Yield"))
        reffig_yield.update_layout(template=plot_settings.dockstreet_template,
                                   plot_bgcolor='white',
                                   hovermode='x',
                                   font_family="Avenir",
                                   font_color="#4c4c4c",
                                   title=dict(font_size=22,
                                              text="<b>10 year bond yield vs NTM earnings yield</b>",
                                              x=0.015,
                                              y=1.,
                                              xref='container',
                                              yref='container'
                                              ),
                                   legend=dict(
                                       orientation="h",
                                       yanchor="bottom",
                                       y=1,
                                       xanchor="left",
                                       x=.0,
                                       font=dict(size=14)
                                   ),
                                   height=500,
                                   margin=dict(t=75)
                                   )
        reffig_yield.update_yaxes(title_text="Yields",
                                  titlefont_size=16,
                                  zeroline=False,
                                  color="#4c4c4c",
                                  tickcolor="#4c4c4c",
                                  tickfont_color="#4c4c4c",
                                  tickfont_size=14,
                                  showgrid=True,
                                  tickformat=",.0",
                                  ticksuffix="%  ",
                                  title_standoff=10,
                                  range=[-.5, df_date_filter_third['ntm_earningsyield'].max()+.25])

        reffig_yield.update_xaxes(tickcolor="#4c4c4c",
                                  tickfont_color="#4c4c4c",
                                  tickfont_size=14,
                                  showgrid=False, )

        for ser in reffig_yield['data']:
            ser['hovertemplate'] = "%{x|%b %-d, %Y}<br>%{y:,.2f}%<extra></extra>"

        st.plotly_chart(reffig_price, use_container_width=True)
        st.plotly_chart(reffig_yield, use_container_width=True)

    # ------------------------------------
    # st.write('<br>', unsafe_allow_html=True)
    # diff_expander = st.beta_expander('Yield Difference / Bond Yield Plot', expanded=False)
    # with diff_expander:
    #     df_date_filter_diff, m_d, u1_d, d1_d, u2_d, d2_d = plot_functions.calc_stds(df,
    #                                                                                  start_date_third,
    #                                                                                  end_date_third,
    #                                                                                  'diff_div_bond')
    #
    #     updated_figure_diff = plot_functions.std_plot_with_button(df_date_filter_diff,
    #                                                                'date',
    #                                                                'diff_div_bond',
    #                                                                'Yield Difference / Bond Yield',
    #                                                                'Plot title for yield difference / bond yield',
    #                                                                m_d,
    #                                                                u1_d,
    #                                                                d1_d,
    #                                                                u2_d,
    #                                                                d2_d,
    #                                                                'diff')
    #
    #     st.plotly_chart(updated_figure_diff, use_container_width=True)

    st.markdown("""
        <style>
        .small-font {
            font-size:12px !important;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="small-font"><br><br>Data from <a href="https://www.factset.com/">FactSet</a></p>',
                unsafe_allow_html=True)


# FOURTH PAGE
def oscillator_page():
    st.title('CoT Oscillator')

    # startindex = len(cftc.date)-1

    comp_col_dict = {'spx':['E-Mini S&P 500 Futures','S&P 500 Price'],
                     'qqq':['E-Mini NASDAQ 100 Futures','QQQ Price'],
                     'dxy':['US Dollar Index Futures','US Dollar Index'],
                     '10y':['US 10y Treasury Note Futures','US 10y Yield'],
                     'wti':['Crude Oil WTI Futures','WTI Price']}

    # choose which sheetname to pull from in cto spreadsheet
    st.sidebar.write('<br><br><b>Oscillator Calculation</b>', unsafe_allow_html=True)
    contract_sel_name = st.sidebar.selectbox(label='Choose contract',
                                        options=['E-Mini S&P 500 Futures','E-Mini NASDAQ 100 Futures',
                                                 'US Dollar Index Futures','US 10y Treasury Note Futures',
                                                 'Crude Oil WTI Futures'],
                                        index=0)
    contract_sel = [k for k,v in comp_col_dict.items() if v[0]==contract_sel_name][0]

    # now that you know which sheetname to use, pull the cto file from google drive
    cftc = pull_cot_data(url_cftc, contract_sel)
    cftc = cftc.sort_values(by='date', ascending=False).reset_index(drop=True)

    # choose number of weeks to look back (not dependent on date range chosen)
    num_wks = st.sidebar.number_input(label='# week lookback',
                                      min_value=4,
                                      max_value=104,
                                      value=52,
                                      step=1,
                                      key='weeks_for_calc')

    # choose which columns to show oscillator for
    st.sidebar.write('<br><b>Choose groups used in oscillators</b>', unsafe_allow_html=True)

    asset_col = st.sidebar.checkbox(label='Asset',
                                    value=True,
                                    key='asset_oscillator'
                                    )

    dealer_col = st.sidebar.checkbox(label='Dealer',
                                    value=False,
                                    key='dealer_oscillator'
                                    )

    leveraged_col = st.sidebar.checkbox(label='Leveraged Funds',
                                    value=False,
                                    key='leverage_oscillator'
                                    )

    levdeal_col = st.sidebar.checkbox(label='Dealer + Leveraged Funds',
                                    value=True,
                                    key='levdeal_oscillator'
                                    )

    # sidebar date filter
    st.sidebar.write('<br>', unsafe_allow_html=True)
    with st.sidebar.form(key='date_form_fourth'):
        st.write('<b>Date Inputs</b>', unsafe_allow_html=True)
        start_date_fourth = st.selectbox(label='Choose a Friday start',
                                         index=208, # 52 weeks * 4 years
                                         options=cftc.date,
                                         format_func=lambda x: x.strftime('%b %d, %Y'),
                                         key='start_fourth')
        end_date_fourth = st.selectbox(label='Choose a Friday end',
                                       index=0,
                                       options=cftc.date,
                                       format_func=lambda x: x.strftime('%b %d, %Y'),
                                       key='end_fourth')
        submit_button_fourth = st.form_submit_button('Submit', help='Press to recalculate')

    # column to use for reference plot for oscillators
    comp_col = cftc.columns.tolist()[4]

    st.subheader(f"{contract_sel_name} vs {comp_col_dict[comp_col][1]}")

    cftc = cftc.sort_values(by='date', ascending=True).reset_index(drop=True)
    cftc = cftc.reset_index(drop=True)

    # date_options = cftc.date.to_frame()
    # start_filter_date = cftc.loc[max(cftc.index[cftc['date'] == start_date_fourth][0]-13,0)]['date']

    metric_col_dict = {'Asset':'asset', 'Dealer':'dealer', 'Leveraged Funds':'leveraged_funds',
                       'Dealer + Leveraged Funds':'lev_deal'}

    # show oscillator plot for each of the groups checked
    if asset_col:
        asset_plot = plot_functions.oscillator_plot(cftc,
                                                    metric_col_dict['Asset'],
                                                    'Asset',
                                                    num_wks,
                                                    start_date_fourth,
                                                    end_date_fourth,
                                                    plot_settings.color_list[3]
                                                   )
        st.plotly_chart(asset_plot, use_container_width=True)

    if dealer_col:
        dealer_plot = plot_functions.oscillator_plot(cftc,
                                                       metric_col_dict['Dealer'],
                                                       'Dealer',
                                                       num_wks,
                                                       start_date_fourth,
                                                       end_date_fourth,
                                                     plot_settings.color_list[0]
                                                   )
        st.plotly_chart(dealer_plot, use_container_width=True)

    if leveraged_col:
        leverage_plot = plot_functions.oscillator_plot(cftc,
                                                       metric_col_dict['Leveraged Funds'],
                                                       'Leveraged Funds',
                                                       num_wks,
                                                       start_date_fourth,
                                                       end_date_fourth,
                                                       plot_settings.color_list[0]
                                                   )
        st.plotly_chart(leverage_plot, use_container_width=True)

    if levdeal_col:
        levdel_plot = plot_functions.oscillator_plot(cftc,
                                                       metric_col_dict['Dealer + Leveraged Funds'],
                                                       'Dealer + Leveraged Funds',
                                                       num_wks,
                                                       start_date_fourth,
                                                       end_date_fourth,
                                                     plot_settings.color_list[0]
                                                   )
        st.plotly_chart(levdel_plot, use_container_width=True)

    # filter to dates chosen and show the reference plot
    cftc = cftc[(cftc['date'] >= start_date_fourth) & (cftc['date'] <= end_date_fourth)]

    compfig = go.Figure()
    compfig.add_trace(go.Scatter(x=cftc['date'],
                                y=cftc[comp_col],
                                line=dict(width=2.25, color=plot_settings.color_list[5]),
                                name=comp_col_dict[comp_col][1]))
    compfig.update_layout(template=plot_settings.dockstreet_template,
                               plot_bgcolor='white',
                               hovermode='x',
                               font_family="Avenir",
                               font_color="#4c4c4c",
                               showlegend=False,
                               height=300,
                               margin=dict(b=25, t=25)
                               )
    compfig.update_yaxes(title_text=f"<b>{comp_col_dict[comp_col][1].upper()}</b><br>Reference Plot",
                          tickformat=",.",
                          titlefont_size=16,
                          zeroline=False,
                          color="#4c4c4c",
                          tickcolor="#4c4c4c",
                          tickfont_color="#4c4c4c",
                          tickfont_size=14,
                          showgrid=True,
                          ticksuffix="  ",
                          title_standoff=10,)
                          # range=[-.05, cftc['kperc'].max() + .025])

    compfig.update_xaxes(tickcolor="#4c4c4c",
                          tickfont_color="#4c4c4c",
                          tickfont_size=14,
                          showgrid=False, )

    for ser in compfig['data']:
        ser['hovertemplate'] = "%{x|%b %-d, %Y}<br>%{y:,.2f}<extra></extra>"

    # st.markdown("""---""")
    st.plotly_chart(compfig, use_container_width=True)




def create_app_with_pages():
    # CREATE PAGES IN APP
    app = MultiApp()
    app.add_app("Earnings Spread Calculation", earnings_recalc)
    app.add_app("S&P Price vs P/E Ratio", something_else)
    app.add_app("Stock to Bond Ratio", earnings_divided_yield)
    app.add_app("CoT Oscillator", oscillator_page)

    app.run()

if __name__ == '__main__':
    create_app_with_pages()