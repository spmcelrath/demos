import bs4 as bs
import pickle
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from os.path import dirname, join
from bokeh.io import curdoc, show, output_file
from bokeh.layouts import column, layout, row, widgetbox, column, Spacer
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, RangeSlider, Slider, MultiChoice, Label, Band, Segment, BooleanFilter, CDSView, Select, DataTable, DateFormatter, TableColumn, NumeralTickFormatter, Panel, Tabs, Dropdown, LegendItem
from bokeh.io import show
from bokeh.models import Button, CheckboxButtonGroup, DatePicker, CustomJS, Toggle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
from bokeh.palettes import Spectral4, linear_palette, turbo
from bokeh.transform import transform    
from bokeh.models import (Plot, Range1d, MultiLine, Circle, HoverTool, BoxSelectTool, ResetTool, CustomJSTransform, LabelSet,
EdgesAndLinkedNodes, NodesAndLinkedEdges, TapTool, StaticLayoutProvider, LassoSelectTool, PanTool, BoxZoomTool, ZoomInTool, ZoomOutTool)
from bokeh.models.widgets import RangeSlider, Button, DataTable, TableColumn, NumberFormatter
from bokeh.plotting import from_networkx
from bokeh.models import CustomJS, DateRangeSlider
from datetime import date, timedelta
from bokeh.transform import factor_cmap
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tornado import gen
from bokeh.document import without_document_lock
from bokeh.models import ColumnDataSource
from bokeh.plotting import curdoc, figure
import datetime
from bokeh.models import Span
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA
from bokeh.palettes import d3
from bokeh.palettes import viridis
from bokeh.layouts import gridplot
import itertools
import math
import pickle
import calendar
from pandas_datareader import data as data_reader
import quantstats as qs
from functools import lru_cache
from bokeh.themes import built_in_themes



doc = curdoc()
itr = 0
executor = ThreadPoolExecutor(max_workers=2)
mds = manifold.MDS(2, dissimilarity='precomputed', metric=True)
animating = False
pc_id_f = 0
selected_names = []
selected_full_names = []
min_corr = 0.75
max_corr = 1.0


sdate = date(2019, 1, 22)   # start date
edate = date(2020, 11, 22)   # end date

delta = edate - sdate       # as timedelta
dates = []
for idx in range(delta.days + 1):
    day = sdate + timedelta(days=idx)
    dates.append(str(day))

def transform_mc(x):
    input_start = 11875423
    input_end = 5577639964
    output_start = 0.010
    output_end = 0.050
    if x > 5577639964:
        return 0.05
    else:
        return (output_start + ((output_end - output_start) / (input_end - input_start)) * (x - input_start))

def transform_width(x):
    input_start = -1
    input_end = 1
    output_start = 0
    output_end = 3
    return (output_start + ((output_end - output_start) / (input_end - input_start)) * (x - input_start))


def transform_color(x):
    testd = [-1, -.5, 0, .5, 1]

    minima = min(testd)
    maxima = max(testd)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm_r)
    rgb = mapper.to_rgba(x)
    return matplotlib.colors.rgb2hex(rgb)

def transform_color_2(df):
    colors = ["#c5001d", "#e23b35", "#f76c52", "#ffb396", "#f4ccb9", "#dddddd", "#c1d5f4", "#a3c5ff", "#85aeff", "#4e6ce3", "#3a48c7"]
    colors = np.array(colors)
    v = df['weight'].values
    v = ((v-v.min())/(v.max()-v.min())*(len(colors)-1)).astype(np.int16)
    return pd.Series(colors[v])


def blocking_task(itr):
    time.sleep(0.001)
    return itr

@lru_cache(maxsize=None)
def recursive_mds(end_date='2020-11-22'):
    if end_date == '2019-01-22':
        corr_mtrx = data[:end_date].tail(60).dropna(axis=1, how='any').corr()
        dist_mtrx = (1 - corr_mtrx)
        dist_mtrx = dist_mtrx.round(4)
        dist_mtrx = dist_mtrx.fillna(0)
        dist_mtrx_arr = np.array(dist_mtrx.values.tolist())
        return mds.fit_transform(dist_mtrx_arr)
    else:
        corr_mtrx = data[:end_date].tail(60).dropna(axis=1, how='any').corr()
        dist_mtrx = (1 - corr_mtrx)
        dist_mtrx = dist_mtrx.round(4)
        dist_mtrx = dist_mtrx.fillna(0)
        dist_mtrx_arr = np.array(dist_mtrx.values.tolist())
        day_before = pd.to_datetime(end_date) - timedelta(days=1)
        day_before = day_before.strftime("%Y-%m-%d")
        return mds.fit_transform(dist_mtrx_arr, init=recursive_mds(end_date=day_before))


@gen.coroutine
def locked_update(itr):
    print(itr)
    iqr_d.location = pd.to_datetime(dates[itr])
    date_picker.value = dates[itr]
    new_coords = recursive_mds(end_date=dates[itr])
    new_x, new_y = list(new_coords[:, 0]), list(new_coords[:, 1])
    ds.data = dict(x=new_x, y=new_y, name=init_names, full_name=init_full_names, size=init_sizes, sector=init_sectors, label_offset=init_offsets)
    fixed_positions = dict(zip(init_names, list(zip(new_x, new_y))))
    graph_renderer.layout_provider.graph_layout = fixed_positions
    

    hs = data[:dates[itr]].tail(60).corr().stack().reset_index()
    hs.columns = ['var1', 'var2','weight']

    hs=hs.loc[(hs['var1'] != hs['var2']) ]
    # graph_renderer.edge_renderer.data_source.data["edge_color"] = transform_color_2(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()])
    new_colors = np.array(transform_color_2(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]))
    # new_colors[np.where(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] < 0.75] =
    mask = np.where((hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] > min_corr) & (hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] < max_corr), True, False)
    # print(len(new_colors), new_colors)
    # print(len(mask), mask)
    new_colors[~mask] = "#1C00ff00"
    graph_renderer.edge_renderer.data_source.data["edge_color"] = new_colors
    print(new_colors)

    hist, edges = np.histogram(hs['weight'], density=True, bins = 'auto')
    hist_df = pd.DataFrame({"corr": hist,
                            "left": edges[:-1],
                            "right": edges[1:]})
    new_hist_src = ColumnDataSource(hist_df)
    hist_src.data = dict(new_hist_src.data)

    hs_select_vs = data[:dates[itr]].tail(60).corr().stack().reset_index()
    hs_select_vs.columns = ['var1', 'var2','weight']

    hs_select_vs=hs_select_vs.loc[(hs_select_vs['var1'] != hs_select_vs['var2']) ]
    hs_select_vs = hs_select_vs.loc[hs_select_vs['var1'].isin(selected_names) | hs_select_vs['var2'].isin(selected_names)]


    hist_select_vs, edges_select_vs = np.histogram(hs_select_vs['weight'], density=True, bins = 'auto')
    hist_select_vs_df = pd.DataFrame({"corr": hist_select_vs,
                            "left": edges_select_vs[:-1],
                            "right": edges_select_vs[1:]})
    new_hist_select_vs_src = ColumnDataSource(hist_select_vs_df)
    hist_select_vs_src.data = dict(new_hist_select_vs_src.data)

    hs_select_intra = data[:dates[itr]].tail(60).corr().stack().reset_index()
    hs_select_intra.columns = ['var1', 'var2','weight']

    hs_select_intra=hs_select_intra.loc[(hs_select_intra['var1'] != hs_select_intra['var2']) ]
    hs_select_intra = hs_select_intra.loc[hs_select_intra['var1'].isin(selected_names) & hs_select_intra['var2'].isin(selected_names)]


    hist_select_intra, edges_select_intra = np.histogram(hs_select_intra['weight'], density=True, bins = 'auto')
    hist_select_intra_df = pd.DataFrame({"corr": hist_select_intra,
                            "left": edges_select_intra[:-1],
                            "right": edges_select_intra[1:]})
    new_hist_select_intra_src = ColumnDataSource(hist_select_intra_df)
    hist_select_intra_src.data = dict(new_hist_select_intra_src.data)

    numstocks = len(selected_names)
    # Create an array of equal weights across all assets
    portfolio_weights_ew = [1 for i in range(0, numstocks)]

    # Calculate the equally-weighted portfolio returns
    # print(returns_df[[sn for sn in selected_names]])
    selected_cols = [str(sn) for sn in selected_names]
    # returns_df = returns_df + 1
    # print(returns_df.head(10))
    returns_df2 = returns_df.copy()
    returns_df2 = returns_df2[pd.to_datetime('2019-01-22'):]
    if numstocks > 0:
        returns_df2.iloc[0, :] = stock_ohlc_df['CCI30'].iloc[0] / numstocks
    # print(returns_df.head(3))
        returns_df2['INDEX'] = returns_df2[selected_cols].cumprod(axis=0).sum(axis = 1)
        # returns_df2['INDEX'] 100*np.exp(np.nan_to_num(df['return'].cumsum()))
        ewp_source.data = dict(x=returns_df2['INDEX'].index, y=returns_df2['INDEX'])
    else:
        ewp_source.data = dict(x=stock_ohlc_df.index, y=stock_ohlc_df['CCI30']*0)

    # temp_edge_colors = [transform_color(temp_corr_mtrx.at[start_node, end_node]) for start_node, end_node, _ in G.edges(data=True)]
    # graph_renderer.edge_renderer.data_source.data['edge_color'] = temp_edge_colors


    # print([k for k in dict(graph_renderer.edge_renderer.data_source.data).keys()])

# this unlocked callback will not prevent other session callbacks from
# executing while it is in flight
@gen.coroutine
@without_document_lock
def unlocked_task():
    global itr
    if itr < (len(dates) - 1):
        itr +=1
        res = yield executor.submit(blocking_task, itr)
        doc.add_next_tick_callback(partial(locked_update, itr=res))
    else:
        print("DONE")

market_caps = {'BTC': 346365975494.4561,
 'ETH': 67583728459.90878,
 'XRP': 21842850222.523933,
 'LINK': 5936469164.92502,
 'LTC': 5857610060.559541,
 'BCH': 5577639964.052779,
 'ADA': 4786566858.959606,
 'BNB': 4454079406.4605255,
 'EOS': 3040178032.936991,
 'XLM': 2327837477.6936054,
 'XMR': 2266260282.143117,
 'TRX': 2166779505.9483557,
 'NEO': 1284117019.9492443,
 'XEM': 1211585019.9538696,
 'MIOTA': 1000208876.2393003,
 'DASH': 914906766.2541883,
 'WAVES': 838141395.4570062,
 'ZEC': 758094203.133095,
 'ETC': 749574136.1409495,
 'OMG': 567756986.1705801,
 'DOGE': 440600446.42333615,
 'BAT': 346071760.6140124,
 'ZRX': 317635516.29863197,
 'DGB': 323742953.14534414,
 'DCR': 277415827.4636605,
 'QTUM': 265219977.84995413,
 'ICX': 233413035.9962433,
 'LRC': 228940951.09181696,
 'KNC': 219378864.69532052,
 'BTG': 170121524.3949117,
 'REP': 170849532.47804466,
 'LSK': 160387532.2448037,
 'ANT': 160524959.35971218,
 'SC': 136195793.77711618,
 'NANO': 133079523.18033548,
 'ZEN': 121503456.32248436,
 'SNT': 122526969.27740875,
 'BNT': 101197544.9164896,
 'GNT': 102673558.68419605,
 'MONA': 93629145.02139471,
 'RLC': 90825502.61457893,
 'XVG': 82637769.25436649,
 'MAID': 79762549.01394832,
 'GNO': 79301216.89833926,
 'STORJ': 71685131.52725092,
 'BTS': 70878700.15251432,
 'STEEM': 64513679.9585957,
 'KMD': 60865604.04901053,
 'CVC': 59606945.2831897,
 'ARDR': 59746645.07842425,
 'MCO': 56495955.73466792,
 'ARK': 48634913.975411475,
 'SYS': 48142832.385533795,
 'HC': 45715135.679752685,
 'XZC': 41763570.42258665,
 'AE': 37146929.74093889,
 'ADX': 33883342.580306016,
 'KIN': 31086446.92811828,
 'MLN': 30901589.671659738,
 'DNT': 29044673.077253997,
 'FUN': 27480155.649692502,
 'BCN': 26625118.135582663,
 'MTL': 21121826.11633633,
 'SALT': 20763697.21186745,
 'PIVX': 20928833.926107936,
 'GBYTE': 17669194.67438208,
 'DGD': 16264690.032434119,
 'GAS': 13741690.462440778,
 'PPT': 13442780.318494057,
 'NXS': 11875423.693942174
 }

sectors = {'BTC': 'Currencies',
 'ETH': 'Smart Contract Platforms',
 'XRP': 'Currencies',
 'LINK': 'Data Management',
 'LTC': 'Currencies',
 'BCH': 'Currencies',
 'ADA': 'Smart Contract Platforms',
 'BNB': 'Centralized Exchanges',
 'EOS': 'Smart Contract Platforms',
 'XLM': 'Currencies',
 'XMR': 'Currencies',
 'TRX': 'Smart Contract Platforms',
 'NEO': 'Smart Contract Platforms',
 'XEM': 'Smart Contract Platforms',
 'MIOTA': 'IoT',
 'DASH': 'Currencies',
 'WAVES': 'Smart Contract Platforms',
 'ZEC': 'Currencies',
 'ETC': 'Smart Contract Platforms',
 'OMG': 'Scaling',
 'DOGE': 'Currencies',
 'BAT': 'Advertising',
 'ZRX': 'Decentralized Exchanges',
 'DGB': 'Currencies',
 'DCR': 'Currencies',
 'QTUM': 'Smart Contract Platforms',
 'ICX': 'Enterprise and BaaS',
 'LRC': 'Decentralized Exchanges',
 'KNC': 'Decentralized Exchanges',
 'BTG': 'Currencies',
 'REP': 'Prediction Markets',
 'LSK': 'Application Development',
 'ANT': 'Misc',
 'SC': 'File Storage',
 'NANO': 'Currencies',
 'ZEN': 'Currencies',
 'SNT': 'Application Development',
 'BNT': 'Decentralized Exchanges',
 'GNT': 'Shared Compute',
 'MONA': 'Currencies',
 'RLC': 'Shared Compute',
 'XVG': 'Currencies',
 'MAID': 'Data Management',
 'GNO': 'Prediction Markets',
 'STORJ': 'File Storage',
 'BTS': 'Smart Contract Platforms',
 'STEEM': 'Content Creation and Distribution',
 'KMD': 'Interoperability',
 'CVC': 'Identity',
 'ARDR': 'Smart Contract Platforms',
 'MCO': 'Payment Platforms',
 'ARK': 'Interoperability',
 'SYS': 'Scaling',
 'HC': 'Interoperability',
 'XZC': 'Currencies',
 'AE': 'Smart Contract Platforms',
 'ADX': 'Advertising',
 'KIN': 'Social Media',
 'MLN': 'Asset Management',
 'DNT': 'Application Development',
 'FUN': 'Gambling',
 'BCN': 'Currencies',
 'MTL': 'Payment Platforms',
 'SALT': 'Lending',
 'PIVX': 'Currencies',
 'GBYTE': 'Smart Contract Platforms',
 'DGD': 'Tokenization',
 'GAS': 'Smart Contract Platforms',
 'PPT': 'Payment Platforms',
 'NXS': 'Enterprise and BaaS'
 }

categories = {'BTC': 'Payments',
 'ETH': 'Infrastructure',
 'XRP': 'Payments',
 'LINK': 'Services',
 'LTC': 'Payments',
 'BCH': 'Payments',
 'ADA': 'Infrastructure',
 'BNB': 'Financial',
 'EOS': 'Infrastructure',
 'XLM': 'Payments',
 'XMR': 'Payments',
 'TRX': 'Infrastructure',
 'NEO': 'Infrastructure',
 'XEM': 'Infrastructure',
 'MIOTA': 'Services',
 'DASH': 'Payments',
 'WAVES': 'Infrastructure',
 'ZEC': 'Payments',
 'ETC': 'Infrastructure',
 'OMG': 'Infrastructure',
 'DOGE': 'Payments',
 'BAT': 'Media and Entertainment',
 'ZRX': 'Financial',
 'DGB': 'Payments',
 'DCR': 'Payments',
 'QTUM': 'Infrastructure',
 'ICX': 'Infrastructure',
 'LRC': 'Financial',
 'KNC': 'Financial',
 'BTG': 'Payments',
 'REP': 'Financial',
 'LSK': 'Infrastructure',
 'ANT': 'Infrastructure',
 'SC': 'Services',
 'NANO': 'Payments',
 'ZEN': 'Payments',
 'SNT': 'Infrastructure',
 'BNT': 'Financial',
 'GNT': 'Services',
 'MONA': 'Payments',
 'RLC': 'Services',
 'XVG': 'Payments',
 'MAID': 'Services',
 'GNO': 'Financial',
 'STORJ': 'Services',
 'BTS': 'Financial',
 'STEEM': 'Media and Entertainment',
 'KMD': 'Infrastructure',
 'CVC': 'Services',
 'ARDR': 'Infrastructure',
 'MCO': 'Payments',
 'ARK': 'Infrastructure',
 'SYS': 'Infrastructure',
 'HC': 'Infrastructure',
 'XZC': 'Payments',
 'AE': 'Infrastructure',
 'ADX': 'Media and Entertainment',
 'KIN': 'Media and Entertainment',
 'MLN': 'Financial',
 'DNT': 'Infrastructure',
 'FUN': 'Media and Entertainment',
 'BCN': 'Payments',
 'MTL': 'Payments',
 'SALT': 'Financial',
 'PIVX': 'Payments',
 'GBYTE': 'Infrastructure',
 'DGD': 'Financial',
 'GAS': 'Infrastructure',
 'PPT': 'Financial',
 'NXS': 'Infrastructure'}

stock_names = {'BTC': 'Bitcoin',
 'ETH': 'Ethereum',
 'XRP': 'Ripple',
 'LINK': 'Chainlink',
 'LTC': 'Litecoin',
 'BCH': 'Bitcoin Cash',
 'ADA': 'Cardano',
 'BNB': 'Binance Coin',
 'EOS': 'EOS',
 'XLM': 'Stellar',
 'XMR': 'Monero',
 'TRX': 'TRON',
 'NEO': 'Neo',
 'XEM': 'NEM',
 'MIOTA': 'IOTA',
 'DASH': 'Dash',
 'WAVES': 'Waves',
 'ZEC': 'Zcash',
 'ETC': 'Ethereum Classic',
 'OMG': 'OMG Network',
 'DOGE': 'Dogecoin',
 'BAT': 'Basic Attention Token',
 'ZRX': '0x',
 'DGB': 'DigiByte',
 'DCR': 'Decred',
 'QTUM': 'Qtum',
 'ICX': 'ICON',
 'LRC': 'Loopring',
 'KNC': 'Kyber Network',
 'BTG': 'Bitcoin Gold',
 'REP': 'Augur',
 'LSK': 'Lisk',
 'ANT': 'Aragon',
 'SC': 'Siacoin',
 'NANO': 'Nano',
 'ZEN': 'Horizen',
 'SNT': 'Status',
 'BNT': 'Bancor',
 'GNT': 'Golem',
 'MONA': 'MonaCoin',
 'RLC': 'iExec RLC',
 'XVG': 'Verge',
 'MAID': 'MaidSafeCoin',
 'GNO': 'Gnosis',
 'STORJ': 'Storj',
 'BTS': 'BitShares',
 'STEEM': 'Steem',
 'KMD': 'Komodo',
 'CVC': 'Civic',
 'ARDR': 'Ardor',
 'MCO': 'MCO',
 'ARK': 'Ark',
 'SYS': 'Syscoin',
 'HC': 'HyperCash',
 'XZC': 'Zcoin',
 'AE': 'Aeternity',
 'ADX': 'AdEx Network',
 'KIN': 'Kin',
 'MLN': 'Melon',
 'DNT': 'district0x',
 'FUN': 'FunFair',
 'BCN': 'Bytecoin',
 'MTL': 'Metal',
 'SALT': 'SALT',
 'PIVX': 'PIVX',
 'GBYTE': 'Obyte',
 'DGD': 'DigixDAO',
 'GAS': 'Gas',
 'PPT': 'Populous',
 'NXS': 'Nexus'}


sectors['SPY'] = 'Index'
market_caps['SPY'] = market_caps['TRX']
stock_names['SPY'] = 'SPX_index'
categories['SPY'] = 'US Equities'
# sectors = get_sp500_sectors()
# stock_names = get_sp500_names()
sectors['CCI30'] = 'Index'
market_caps['CCI30'] = market_caps['TRX']
stock_names['CCI30'] = 'CCi30_index'
categories['CCI30'] = 'Mid Cap Crypto'

sectors['BITX'] = 'Index'
market_caps['BITX'] = market_caps['TRX']
stock_names['BITX'] = 'BITX_index'
categories['BITX'] = 'Large Cap Crypto '

sectors['BITW20'] = 'Index'
market_caps['BITW20'] = market_caps['TRX']
stock_names['BITW20'] = 'BITW20_index'
categories['BITW20'] = 'Mid Cap Crypto'

sectors['BITW70'] = 'Index'
market_caps['BITW70'] = market_caps['TRX']
stock_names['BITW70'] = 'BITW70_index'
categories['BITW70'] = 'Small Cap Crypto'

sectors['BITW100'] = 'Index'
market_caps['BITW100'] = market_caps['TRX']
stock_names['BITW100'] = 'BITW100_index'
categories['BITW100'] = 'Total Market Crypto'

index_names = {
  'VTI': 'Total US Stock Market',
  'RTY': 'US Small Cap',
  'VNQ': 'US Real Estate',
  'VEA': 'Intl. Developed Markets', 
  'VWO': 'Intl. Emerging Markets', 
  'FM': 'Frontier Markets', 
  'VTIP': 'TIPS', 
  'BND': 'Total Bonds', 
  'JNK': 'High Yield Bonds', 
  'DBC': 'Broad Commodities',
  'GLD': 'Gold', 
  'BIZD': 'BDC Income', 
  'UUP': 'USD'
}

for symbol in index_names:
    sectors[symbol] = 'Index'
    categories[symbol] = index_names[symbol]
    stock_names[symbol] = index_names[symbol]
    market_caps[symbol] = market_caps['TRX']




stock_ohlc_df = pd.read_csv('https://raw.githubusercontent.com/spmcelrath/demos/main/app/stock-corrs/all_data.csv')
stock_ohlc_df = stock_ohlc_df.set_index('Date')
stock_ohlc_df = stock_ohlc_df.drop(['USDT'], axis=1)


ohlc_df = stock_ohlc_df
ohlc_df.index = pd.to_datetime(ohlc_df.index)
data = ohlc_df.pct_change()


data = data.fillna(method='bfill')
data.index = pd.to_datetime(data.index)


init_names = data.columns
init_sizes = [transform_mc(market_caps[n]) for n in init_names]
init_offsets = [s*1000 for s in init_sizes]
init_sectors = [sectors[n] if n in sectors else 'Misc' for n in init_names]
init_categories = [categories[n] if n in categories else 'Misc' for n in init_names]
init_full_names = [stock_names[n] if n in stock_names else 'NA' for n in init_names]
init_coords = recursive_mds(end_date='2019-01-22')
init_x, init_y = list(init_coords[:, 0]), list(init_coords[:, 1])
print(data)
print(init_names)
print(init_sizes)
print(init_offsets)
print(init_sectors)
print(init_full_names)
print(init_coords)




hs = data[:dates[itr]].tail(60).corr().stack().reset_index()
hs.columns = ['var1', 'var2','weight']

hs=hs.loc[(hs['var1'] != hs['var2']) ]



hist, edges = np.histogram(hs['weight'], density=True, bins = 'auto')
hist_df = pd.DataFrame({"corr": hist,
                        "left": edges[:-1],
                        "right": edges[1:]})
hist_src = ColumnDataSource(hist_df)


hs_select_vs = data[:dates[itr]].tail(60).corr().stack().reset_index()
hs_select_vs.columns = ['var1', 'var2','weight']

hs_select_vs=hs_select_vs.loc[(hs_select_vs['var1'] != hs_select_vs['var2']) ]
hs_select_vs = hs_select_vs.loc[hs_select_vs['var1'].isin([]) | hs_select_vs['var2'].isin([])]


hist_select_vs, edges_select_vs = np.histogram(hs_select_vs['weight'], density=True, bins = 'auto')
hist_select_vs_df = pd.DataFrame({"corr": hist_select_vs,
                        "left": edges_select_vs[:-1],
                        "right": edges_select_vs[1:]})
hist_select_vs_src = ColumnDataSource(hist_select_vs_df)

hs_select_intra = data[:dates[itr]].tail(60).corr().stack().reset_index()
hs_select_intra.columns = ['var1', 'var2','weight']

hs_select_intra=hs_select_intra.loc[(hs_select_intra['var1'] != hs_select_intra['var2']) ]
hs_select_intra = hs_select_intra.loc[hs_select_intra['var1'].isin([]) & hs_select_intra['var2'].isin([])]


hist_select_intra, edges_select_intra = np.histogram(hs_select_intra['weight'], density=True, bins = 'auto')
hist_select_intra_df = pd.DataFrame({"corr": hist_select_intra,
                        "left": edges_select_intra[:-1],
                        "right": edges_select_intra[1:]})
hist_select_intra_src = ColumnDataSource(hist_select_intra_df)


csdate = date(2019, 1, 22)   # start date
cedate = date(2020, 11, 22)   # end date

delta = cedate - csdate       # as timedelta
corr_dates = []
for idx in range(delta.days + 1):
    day = csdate + timedelta(days=idx)
    corr_dates.append(str(day))

med_corrs = {}
q1 = {}
q3 = {}

for d in corr_dates:
    # print(ohlc_df)
    corr_df = ohlc_df.pct_change()[:pd.to_datetime(d)].tail(60).dropna(axis=1, how='any').corr()
    # print(corr_df)
    # print(corr_df.stack().reset_index())
    med_corrs[d] = corr_df.stack().reset_index()[0].quantile(0.50)
    q1[d] = corr_df.stack().reset_index()[0].quantile(0.25)
    q3[d] = corr_df.stack().reset_index()[0].quantile(0.75)


    
    
    # cs = df.pct_change()[:pd.to_datetime(d)].tail(60).corr().fillna(0)
    # med_corr = np.median(cs)
    # med_corrs[d] = med_corr



mcs = pd.Series(med_corrs)
mcs.index = pd.to_datetime(mcs.index)
mcs = mcs[pd.to_datetime('2019-01-22'):]

q1s = pd.Series(q1)
q1s.index = pd.to_datetime(q1s.index)
q1s = q1s[pd.to_datetime('2019-01-22'):]

q3s = pd.Series(q3)
q3s.index = pd.to_datetime(q3s.index)
q3s = q3s[pd.to_datetime('2019-01-22'):]

band_data = []

for i, val in mcs.iteritems():
    band_data.append([i, q1s[i], val, q3s[i]])

band_df = pd.DataFrame(band_data, columns=['date', 'q1', 'med', 'q3'])
band_source = ColumnDataSource(band_df)

# band_df = band_df.set_index('date')


# print(mcs[pd.to_datetime('2020-11-22'):])
iqr_d = Span(location=pd.to_datetime(dates[itr]),
                              dimension='height', line_color='grey',
                              line_width=2, line_alpha=0.3)

# pos = nx.spring_layout(G,pos=fixed_positions, fixed = init_names)

# for i in range(0, len(init_x), 2):
#     print("x:", init_x[i:i+2])
#     print("y:", init_y[i:i+2])

pal_len = len(pd.Series(init_categories).unique())
new_pal = turbo(pal_len)
init_colors=factor_cmap('category', new_pal, pd.Series(init_categories).unique(), nan_color='black') 


ds = ColumnDataSource(dict(x=init_x, y=init_y, name=init_names, full_name=init_full_names, sector=init_sectors, category=init_categories, size=init_sizes, label_offset=init_offsets))

returns_df = ohlc_df.pct_change().fillna(0)
returns_df.index = pd.to_datetime(returns_df.index)

returns_df = returns_df + 1


returns_df = returns_df[pd.to_datetime('2019-01-22'):]
ewp_source = ColumnDataSource(dict(x=stock_ohlc_df.index, y=stock_ohlc_df['CCI30']*0))

stats_data = dict(
        metrics=['Cummulative Return', 'Sharpe', 'Sortino', 'Max Drawdown', 'Volatility'],
        benchmark=[
            str(round(qs.stats.comp(stock_ohlc_df['CCI30'][pd.to_datetime('2019-01-22'):].pct_change())*100, 4)) + '%',
            round(qs.stats.sharpe(stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].pct_change()), 4),
            round(qs.stats.sortino(stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].pct_change()), 4),
            str(round(qs.stats.max_drawdown(stock_ohlc_df['CCI30'][pd.to_datetime('2019-01-22'):])*100, 4)) + '%',
            round(qs.stats.volatility(stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].pct_change(), annualize=False), 4)
            ],
            selected=[
            str(round(0, 4)) + '%',
            round(0, 4),
            round(0, 4),
            str(round(0, 4)) + '%',
            round(0, 4)
            ]
)
stats_source = ColumnDataSource(stats_data)

stats_columns = [
        TableColumn(field="metrics", title="metrics"),
        TableColumn(field="selected", title="selection"),
        TableColumn(field="benchmark", title="index"),
    ]
stats_table = DataTable(source=stats_source, columns=stats_columns, width=400, height=300, index_position=None)


plot = figure(plot_width=1100, plot_height=1000, x_range=(-1.0,1.0), y_range=(-1.0,1.0),
              toolbar_location="below", background_fill_color="#fafafa")
plot.axis.visible = False
plot.xgrid.visible = False
plot.ygrid.visible = False

labels = LabelSet(x='x', y='y', text='name', level='overlay',
              x_offset=0, y_offset='label_offset', source=ds, render_mode='canvas', text_color='grey')
plot.add_layout(labels)

hist_p = figure(plot_height=200, plot_width=420, x_range=(-1.0, 1.0), tools='', background_fill_color="#fafafa", toolbar_location=None, margin=(5, 5, 5, 5))
hist_p.quad(bottom = 0, top = "corr",left = "left", right = "right", source = hist_src,  fill_color="#53b497", line_color="#53b497", fill_alpha=0.8, line_alpha=0.5, legend_label="All assets")
hist_p.quad(bottom = 0, top = "corr",left = "left", right = "right", source = hist_select_vs_src,  fill_color="purple", line_color="purple", fill_alpha=0.4, line_alpha=0.5, legend_label="Selection vs. others", level='overlay')
hist_p.quad(bottom = 0, top = "corr",left = "left", right = "right", source = hist_select_intra_src,  fill_color="orange", line_color="orange", fill_alpha=0.2, line_alpha=0.5, legend_label="Intra-Selection", level='overlay')
hist_p.yaxis.visible = False
hist_p.grid.grid_line_alpha=0.3
hist_p.legend.border_line_width = 0
hist_p.legend.border_line_color = None
hist_p.legend.background_fill_color = None
hist_p.legend.background_fill_alpha = 0.0
hist_p.legend.location = "top_left"
hist_p.legend.title = "Frequency distribution of correlations:"
hist_p.legend.label_text_font_size = '8pt'


ewi_leg_label = "equal-weighted index of selection (rebased)"

ewi_p = figure(plot_height=180, plot_width=420,  x_range=(pd.to_datetime('2019-01-22'), pd.to_datetime('2020-11-22')), x_axis_type="datetime", tools='', background_fill_color="#fafafa", toolbar_location=None, y_axis_location="right", margin=(5, 5, 5, 5), y_axis_type='log')
ewi_p.line('x', 'y', color='orange', legend_label=ewi_leg_label, line_alpha=0.8, source=ewp_source)

ewi_p.yaxis.minor_tick_line_color = None
ewi_p.legend.location = "top_left"
ewi_p.legend.border_line_width = 0
ewi_p.legend.border_line_color = None
ewi_p.legend.background_fill_color = None
ewi_p.legend.background_fill_alpha = 0.0
ewi_p.grid.grid_line_alpha=0.3
ewi_p.legend.label_text_font_size = '8pt'


iqr_p = figure(plot_height=200, plot_width=420, x_range=(pd.to_datetime('2019-01-22'), pd.to_datetime('2020-11-22')), y_range=(-0.2,1.0), x_axis_type="datetime", tools='', background_fill_color="#fafafa", toolbar_location=None, y_axis_location="right", margin=(5, 5, 5, 5))
iqr_p.line(x='date', y='med', color='#53b497', legend_label='Median correl.', line_width=2, source=band_source)
band = Band(base='date', lower='q1', upper='q3', source = band_source,
            level='underlay', fill_color='#53b497', fill_alpha=0.2, line_width=1, line_color='#53b497', line_alpha=0.4)
iqr_p.add_layout(band)
iqr_p.xaxis.visible = False
iqr_p.legend.location = "top_left"
iqr_p.legend.border_line_width = 0
iqr_p.legend.border_line_color = None
iqr_p.legend.background_fill_color = None
iqr_p.legend.background_fill_alpha = 0.0
iqr_p.yaxis.minor_tick_line_color = None
iqr_p.grid.grid_line_alpha=0.3
iqr_p.legend.label_text_font_size = '8pt'

index_source = ColumnDataSource(dict(x=ohlc_df.index, y=ohlc_df['CCI30']))

index_p = figure(plot_height=200, plot_width=420, x_range=(pd.to_datetime('2019-01-22'), pd.to_datetime('2020-11-22')), x_axis_type="datetime", tools='box_select', background_fill_color="#fafafa", toolbar_location=None, y_axis_location="right", margin=(5, 5, 5, 5), y_axis_type='log')
index_line = index_p.line(x='x', y='y', color='grey', line_alpha=0.8, source=index_source)

index_p.yaxis.minor_tick_line_color = None
index_p.legend.location = "top_left"
index_p.legend.border_line_width = 0
index_p.legend.border_line_color = None
index_p.legend.background_fill_color = None
index_p.legend.background_fill_alpha = 0.0
index_p.grid.grid_line_alpha=0.3
index_p.xaxis.visible = False
index_p.legend.label_text_font_size = '8pt'

iqr_p.add_layout(iqr_d)
index_p.add_layout(iqr_d)
ewi_p.add_layout(iqr_d)

plot.legend.border_line_width = 0
plot.legend.border_line_color = 'grey'
plot.legend.border_line_alpha = 0.2
plot.legend.background_fill_color = 'grey'
plot.legend.background_fill_alpha = 0.2



af_button = Button(label="animate forward >>", button_type="success")
ab_button = Button(label="<< animate backward", button_type="default")

def select_callback(attr, old, new):
    global selected_names
    global selected_full_names
    selected_names = []
    selected_full_names = []

    for i in new:
        selected_names.append(ds.data['name'][i])
        selected_full_names.append(ds.data['full_name'][i])
    # selected_str = ', '.join([str(elem) for elem in selected_full_names]) 
    # selected_label_txt = "Selected: {}".format(selected_str)
    # selected_label.text = selected_label_txt
    
    
    hs_select_vs = data[:dates[itr]].tail(60).corr().stack().reset_index()
    hs_select_vs.columns = ['var1', 'var2','weight']

    hs_select_vs=hs_select_vs.loc[(hs_select_vs['var1'] != hs_select_vs['var2']) ]
    hs_select_vs = hs_select_vs.loc[hs_select_vs['var1'].isin(selected_names) | hs_select_vs['var2'].isin(selected_names)]


    hist_select_vs, edges_select_vs = np.histogram(hs_select_vs['weight'], density=True, bins = 'auto')
    hist_select_vs_df = pd.DataFrame({"corr": hist_select_vs,
                            "left": edges_select_vs[:-1],
                            "right": edges_select_vs[1:]})
    new_hist_select_vs_src = ColumnDataSource(hist_select_vs_df)
    hist_select_vs_src.data = dict(new_hist_select_vs_src.data)

    hs_select_intra = data[:dates[itr]].tail(60).corr().stack().reset_index()
    hs_select_intra.columns = ['var1', 'var2','weight']

    hs_select_intra=hs_select_intra.loc[(hs_select_intra['var1'] != hs_select_intra['var2']) ]
    hs_select_intra = hs_select_intra.loc[hs_select_intra['var1'].isin(selected_names) & hs_select_intra['var2'].isin(selected_names)]


    hist_select_intra, edges_select_intra = np.histogram(hs_select_intra['weight'], density=True, bins = 'auto')
    hist_select_intra_df = pd.DataFrame({"corr": hist_select_intra,
                            "left": edges_select_intra[:-1],
                            "right": edges_select_intra[1:]})
    new_hist_select_intra_src = ColumnDataSource(hist_select_intra_df)
    hist_select_intra_src.data = dict(new_hist_select_intra_src.data)

    numstocks = len(selected_names)
    # print(numstocks)

    # Create an array of equal weights across all assets
    portfolio_weights_ew = [1 for i in range(0, numstocks)]

    # Calculate the equally-weighted portfolio returns
    # print(returns_df[[sn for sn in selected_names]])
    selected_cols = [str(sn) for sn in selected_names]
    # print(selected_cols)
    # for sc in selected_cols:
    returns_df2 = returns_df.copy()

    returns_df2 = returns_df2[pd.to_datetime('2019-01-22'):]
    if numstocks > 0:
        returns_df2.iloc[0, :] = stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].iloc[0] / numstocks
    # print(returns_df.head(3))
        returns_df2['INDEX'] = returns_df2[selected_cols].cumprod(axis=0).sum(axis = 1)
        ewp_source.data = dict(x=returns_df2['INDEX'].index, y=returns_df2['INDEX'])
        stats_data = dict(
        metrics=['Cummulative Return', 'Sharpe', 'Sortino', 'Max Drawdown', 'Volatility'],
        benchmark=[
            str(round(qs.stats.comp(stock_ohlc_df['CCI30'][pd.to_datetime('2019-01-22'):].pct_change())*100, 4)) + '%',
            round(qs.stats.sharpe(stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].pct_change()), 4),
            round(qs.stats.sortino(stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].pct_change()), 4),
            str(round(qs.stats.max_drawdown(stock_ohlc_df['CCI30'][pd.to_datetime('2019-01-22'):])*100, 4)) + '%',
            round(qs.stats.volatility(stock_ohlc_df[pd.to_datetime('2019-01-22'):]['CCI30'].pct_change(), annualize=False), 4)
            ],
        selected=[
            str(round(qs.stats.comp(returns_df2['INDEX'].pct_change())*100, 4)) + '%',
            round(qs.stats.sharpe(returns_df2['INDEX'].pct_change()), 4),
            round(qs.stats.sortino(returns_df2['INDEX'].pct_change()), 4),
            str(round(qs.stats.max_drawdown(returns_df2['INDEX'])*100, 4)) + '%',
            round(qs.stats.volatility(returns_df2['INDEX'].pct_change(), annualize=False), 4)
        ])
        stats_source.data = stats_data
        
    else:
        ewp_source.data = dict(x=stock_ohlc_df.index, y=stock_ohlc_df['CCI30']*0)

ds.selected.on_change('indices', select_callback)


def af_callback(attr):
    global animating
    global pc_id_f
    
    if af_button.label == 'animate forward >>':
        af_button.label = 'pause ||'
    else:
        af_button.label = 'animate forward >>'
    if af_button.button_type == "success":
        af_button.button_type='primary'
    else:
        af_button.button_type='success'

    if not animating:
        pc_id_f = doc.add_periodic_callback(unlocked_task, 200)
        animating = True
    else:
        doc.remove_periodic_callback(pc_id_f)
        animating = False

af_button.on_click(af_callback)

LABELS = ["legend", "symbols"]

checkbox_button_group = CheckboxButtonGroup(labels=LABELS, active=[0])
labels.visible = False
def check_btn_callback(attr):
    if 1 in attr:
        print("SYMBOLS ON")
        labels.visible = True
    else:
        print("SYMBOLS OFF")
        labels.visible = False
    if 0 in attr:
        print("LEGEND ON")
        plot.legend.visible=True
    else:
        print("LEGEND OFF")
        plot.legend.visible=False

checkbox_button_group.on_click(check_btn_callback)



date_picker = DatePicker(value="2019-01-22", min_date="2019-01-22", max_date="2020-11-22")

as_slider = Slider(start=1, end=10, value=5, step=.1, title="animation speed")
l_slider = RangeSlider(start=-1.00, end=1.00, value=(0.75, 1.0), step=0.01, title="show links for")
spacer = Spacer(width=240, height=700)
controls = [spacer, date_picker, af_button, ab_button, as_slider, l_slider, checkbox_button_group]

index_select = Select(title="Benchmark:", value="CCI30 (The Cryptocurrencies Index)", options=["CCI30 (The Cryptocurrencies Index)", "BITX (Bitwise 10 Large Cap Crypto)", "BITW20 (Bitwise 20 Mid Cap Crypto)", "BITW70 (Bitwise 70 Small Cap Crypto)", "BITW100 (Bitwise Total Market Crypto)"], width=400)

def is_callback(attr, old, new):
    symbol = new.split(' ')[0]
    index_source.data = dict(x=ohlc_df.index, y=ohlc_df[symbol])


index_select.on_change('value', is_callback)

def ls_callback(attr, old, new):
    global min_corr
    global max_corr

    print(old, new)
    min_corr = new[0]
    max_corr = new[1]
    new_colors = np.array(transform_color_2(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]))
    # new_colors[np.where(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] < 0.75] =
    mask = np.where((hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] > min_corr) & (hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] < max_corr), True, False)
    # print(len(new_colors), new_colors)
    # print(len(mask), mask)
    new_colors[~mask] = "#1C00ff00"
    graph_renderer.edge_renderer.data_source.data["edge_color"] = new_colors
    # new_corr_bools = [True if ((corr_val < new[1]) and (corr_val > new[0])) else False for corr_val in segment_source.data['r']]
    # new_corr_view = CDSView(source=segment_source, filters=[BooleanFilter(corr_bools)])
    # corr_view.filters[0] = BooleanFilter(new_corr_bools)
    # plot_segment.view = new_corr_view
    # plot_segment.update()

l_slider.on_change('value_throttled', ls_callback)


# plot_scatter.visible = False

G = nx.Graph()
G.add_nodes_from(init_names)
G.add_edges_from(itertools.combinations(init_names, 2))
fixed_positions = dict(zip(init_names, list(zip(init_x, init_y))))
# 
graph_renderer = from_networkx(G, nx.spring_layout, pos=fixed_positions, fixed=init_names, center=(0,0))
node_attrs = {}

for i, name in enumerate(init_names):
        node_attrs[name] = {}
        node_attrs[name]['size'] = init_sizes[i]
        node_attrs[name]['sector'] = init_sectors[i]
        node_attrs[name]['name'] = init_names[i]
        node_attrs[name]['full_name'] = init_full_names[i]
        node_attrs[name]['category'] = init_categories[i]




nx.set_node_attributes(G, node_attrs)

graph_renderer.node_renderer.data_source.data['sector'] = [G.nodes[n]['sector'] for n in G.nodes()]
graph_renderer.node_renderer.data_source.data['category'] = [G.nodes[n]['category'] for n in G.nodes()]
graph_renderer.node_renderer.data_source.data['size'] = [G.nodes[n]['size']*1000 for n in G.nodes()]
graph_renderer.node_renderer.data_source.data['name'] = [G.nodes[n]['name'] for n in G.nodes()]
graph_renderer.node_renderer.data_source.data['full_name'] = [G.nodes[n]['full_name'] for n in G.nodes()]


corr_mtrx = data[:'2019-01-22'].tail(60).dropna(axis=1, how='any').corr()
# weight_map = dict(zip(zip(source, target), weight))
edge_color_attrs = {}
edge_width_attrs = {}
edge_weight_attrs = {}

for start_node, end_node, _ in G.edges(data=True):
    edge_corr = corr_mtrx.at[start_node, end_node]
    edge_color = transform_color(edge_corr)
    edge_width = transform_width(edge_corr)
    edge_weight = edge_corr

    edge_color_attrs[(start_node, end_node)] = edge_color
    edge_width_attrs[(start_node, end_node)] = edge_width
    edge_weight_attrs[(start_node, end_node)] = edge_weight



nx.set_edge_attributes(G, edge_color_attrs, "edge_color")
nx.set_edge_attributes(G, edge_width_attrs, "edge_width")
nx.set_edge_attributes(G, edge_weight_attrs, "edge_weight")

new_colors = np.array(transform_color_2(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]))
# new_colors[np.where(hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] < 0.75] =
mask = np.where((hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] > min_corr) & (hs[~hs[['var1', 'var2']].apply(frozenset, axis=1).duplicated()]['weight'] < max_corr), True, False)
# print(len(new_colors), new_colors)
# print(len(mask), mask)
new_colors[~mask] = "#1C00ff00"
graph_renderer.edge_renderer.data_source.data["edge_color"] = new_colors

# graph_renderer.edge_renderer.data_source.data['edge_color'] = [_['edge_color'] for start_node, end_node, _ in G.edges(data=True)]
graph_renderer.edge_renderer.data_source.data['edge_width'] = [_['edge_width'] for start_node, end_node, _ in G.edges(data=True)]
graph_renderer.edge_renderer.data_source.data['edge_weight'] = [_['edge_weight'] for start_node, end_node, _ in G.edges(data=True)]

# corr_bools = np.where(graph_renderer.edge_renderer.data_source.data['edge_weight'].values > 0.75, True, False)
# # corr_bools = [True if corr_val > 0.75 else False for corr_val in graph_renderer.edge_renderer.data_source.data['edge_weight'].values]

# corr_view = CDSView(source=graph_renderer.edge_renderer.data_source, filters=[BooleanFilter(corr_bools)])

# graph_renderer.edge_renderer.view = corr_view

graph_renderer.edge_renderer.glyph = MultiLine(line_color=None, line_alpha=0.0)
graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color="edge_color", line_width="edge_width", line_alpha=1.0)
graph_renderer.edge_renderer.level = 'underlay'
graph_renderer.node_renderer.level = 'glyph'
graph_renderer.node_renderer.glyph = Circle(
    size='size', 
    fill_color=factor_cmap(
        'sector', 
        new_pal, 
        pd.Series(init_sectors).unique(), 
    nan_color='black'
    ), 
    fill_alpha=0.0,
    line_alpha=0.0)

graph_renderer.node_renderer.selection_glyph = Circle(
    size='size', 
    fill_color=factor_cmap(
        'sector', 
        new_pal, 
        pd.Series(init_sectors).unique(), 
    nan_color='black'
    ), 
    fill_alpha=0.0,
    line_alpha=0.0
    )

graph_renderer.selection_policy = NodesAndLinkedEdges()
# graph_renderer.inspection_policy = EdgesAndLinkedNodes()

plot.renderers.append(graph_renderer)

plot_scatter = plot.scatter(
    source= ds,
    x='x', 
    y='y',
    radius='size',
    fill_alpha=0.5,
    fill_color=init_colors,
    line_alpha=0.8,
    line_color=init_colors,
    selection_fill_alpha=0.9, 
    selection_line_alpha=0.9,
    selection_line_color='black', 
    selection_line_width=2,
    nonselection_fill_alpha=0.3,
    nonselection_line_alpha=0.3,
    legend_group='category'
)

# legend_order = [2, 6, 10, 12, 13, 0, 1, 3, 4, 5, 7, 8, 9, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
# plot.legend[0].items = [plot.legend[0].items[i] for i in legend_order]



legend_order = [2, 6, 10, 12, 13, 9, 11, 14, 17, 0, 1, 3, 4, 5, 7, 8, 15, 16,  18, 19, 20, 21]
plot.legend[0].items = [plot.legend[0].items[i] for i in legend_order]

plot.legend[0].items.insert(0, LegendItem(label='Cryptoassets Categories:'))

plot.legend[0].items.insert(6, LegendItem(label=''))

plot.legend[0].items.insert(7, LegendItem(label='Indices/Benchmarks:'))




for li in plot.legend[0].items:
    print(li.label)

plot.legend.background_fill_alpha = 0.5
plot.legend.location = "bottom_right"
main_plot = row(plot, width=1100, height=1000)
main_plot.sizing_mode = "fixed"
inputs = column(*controls, width=240, height=1000)
inputs.sizing_mode = "fixed"
plots = column([hist_p,  iqr_p, index_select, index_p, ewi_p, stats_table], width=450, height=1000)
plots.sizing_mode = "fixed"

print(stock_ohlc_df.isnull().values.any())
print(ohlc_df.isnull().values.any())
print(data.isnull().values.any())
print(hs.isnull().values.any())
print(returns_df.isnull().values.any())

print(stock_ohlc_df.dtypes)
print(ohlc_df.dtypes)
print(data.dtypes)
print(hs.dtypes)
print(returns_df.dtypes)



TOOLTIPS = [
    ("symbol", "@name"),
    ("name", "@full_name"),
    ("category", "@category"),
    ("Sector", "@sector")


]
plot.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[graph_renderer]), TapTool(), LassoSelectTool())

l = layout([

    [inputs, main_plot, plots]
    
], sizing_mode="scale_both")

doc.add_root(l)
