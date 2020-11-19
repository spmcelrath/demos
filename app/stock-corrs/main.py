import bs4 as bs
import pickle
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from os.path import dirname, join
from bokeh.io import curdoc, show, output_file
from bokeh.layouts import column, layout, row, widgetbox, column, Spacer
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, RangeSlider, Slider, MultiChoice, Label, Band, Segment, BooleanFilter, CDSView, Select, DataTable, DateFormatter, TableColumn, NumeralTickFormatter, Panel, Tabs
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

sdate = date(2015, 3, 1)   # start date
edate = date(2020, 10, 30)   # end date

delta = edate - sdate       # as timedelta
dates = []
for idx in range(delta.days + 1):
    day = sdate + timedelta(days=idx)
    dates.append(str(day))

def transform_mc(x):
    input_start = 3315097344
    input_end = 2017008549888
    output_start = 0.005
    output_end = 0.050
    if x > input_end:
        return 0.090
    else:
        return (output_start + ((output_end - output_start) / (input_end - input_start)) * (x - input_start))

def blocking_task(itr):
    time.sleep(0.001)
    return itr

@lru_cache(maxsize=None)
def recursive_mds(end_date='2020-01-01'):
    if end_date == '2015-03-01':
        corr_mtrx = data[:end_date].tail(65).dropna(axis=1, how='any').corr()
        dist_mtrx = 1 - corr_mtrx
        dist_mtrx = dist_mtrx.round(4)
        dist_mtrx = dist_mtrx.fillna(0)
        dist_mtrx_arr = np.array(dist_mtrx.values.tolist())
        return mds.fit_transform(dist_mtrx_arr)
    else:
        corr_mtrx = data[:end_date].tail(65).dropna(axis=1, how='any').corr()
        dist_mtrx = 1 - corr_mtrx
        dist_mtrx = dist_mtrx.round(4)
        dist_mtrx = dist_mtrx.fillna(0)
        dist_mtrx_arr = np.array(dist_mtrx.values.tolist())
        day_before = pd.to_datetime(end_date) - timedelta(days=1)
        day_before = day_before.strftime("%Y-%m-%d")
        return mds.fit_transform(dist_mtrx_arr, init=recursive_mds(end_date=day_before))


@gen.coroutine
def locked_update(itr):
    print(itr)
    new_coords = recursive_mds(end_date=dates[itr])
    new_x, new_y = list(new_coords[:, 0]), list(new_coords[:, 1])
    ds.data = dict(x=new_x, y=new_y, name=init_names, full_name=init_full_names, size=init_sizes, sector=init_sectors, label_offset=init_offsets)

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



market_caps = {'AAPL': 2017008549888, 'MSFT': 1634731425792, 'AMZN': 1553581735936, 'GOOGL': 1197932019712, 'FB': 783261499392, 'V': 459052580864, 'WMT': 425657597952, 'JNJ': 393538404352, 'PG': 356369563648, 'JPM': 348028239872, 'UNH': 337751801856, 'MA': 331298111488, 'NVDA': 324139220992, 'HD': 310802808832, 'VZ': 252423487488, 'DIS': 249690521600, 'BAC': 234739187712, 'KO': 228000153600, 'CRM': 226235105280, 'CMCSA': 224579584000, 'ADBE': 224215875584,  'PFE': 212247511040, 'NFLX': 211655098368, 'T': 205798178816, 'MRK': 204843876352, 'NKE': 201055977472, 'ABT': 199036026880, 'PEP': 198864027648, 'TMO': 193712701440, 'INTC': 185413992448, 'ABBV': 174735818752, 'CSCO': 174686896128, 'ORCL': 171319640064, 'COST': 166651822080, 'DHR': 166434226176, 'QCOM': 162581266432, 'CVX': 159910576128, 'TMUS': 158437900288, 'MCD': 158388240384, 'XOM': 152660230144, 'ACN': 152627134464, 'NEE': 151318560768, 'MDT': 150762127360, 'AVGO': 150568812544, 'BMY': 143256846336, 'TXN': 142914387968, 'UPS': 141553811456, 'HON': 141109067776, 'AMGN': 138521296896, 'UNP': 136842584064, 'LLY': 136073789440, 'LIN': 134788521984, 'CHTR': 129243176960, 'LOW': 120207368192, 'PM': 118114934784, 'SBUX': 111551488000, 'AMT': 107281874944, 'BA': 105143771136, 'LMT': 104625233920, 'IBM': 104013086720, 'MS': 102710476800, 'C': 101820211200, 'BLK': 101324185600, 'RTX': 99992518656, 'WFC': 99393142784, 'MMM': 97869389824, 'NOW': 97745100800, 'AMD': 97359372288, 'INTU': 93232087040, 'CAT': 93182320640, 'CVS': 92803153920, 'AXP': 92473425920, 'EL': 92155297792, 'FIS': 89390522368, 'ISRG': 88038703104, 'SYK': 86852812800, 'SCHW': 85356003328, 'BKNG': 84057882624, 'ANTM': 82651799552, 'MDLZ': 82134089728, 'SPGI': 81970012160, 'TGT': 81495597056, 'GE': 80459415552, 'CI': 79886974976, 'DE': 78769438720, 'ZTS': 78352678912, 'PLD': 76343255040, 'GILD': 76296101888, 'GS': 75794735104, 'MO': 74689896448, 'ADP': 73460129792, 'CL': 72953659392, 'CCI': 72235122688, 'FISV': 72065384448, 'FDX': 71556317184, 'TJX': 71296106496, 'BDX': 70145196032, 'DUK': 70041214976, 'D': 69589360640, 'CSX': 69032411136, 'EQIX': 67775545344, 'SO': 67477889024, 'CB': 66989486080, 'ITW': 66883842048, 'SHW': 65720684544, 'AMAT': 65601118208, 'USB': 65356894208, 'MU': 64449871872, 'TFC': 63873929216, 'LRCX': 61424156672, 'NSC': 60463665152, 'REGN': 60247273472, 'ATVI': 59722526720, 'CME': 59529682944, 'ECL': 59309760512, 'VRTX': 58723082240, 'GM': 58597826560, 'APD': 58094014464, 'HUM': 57427390464, 'MMC': 57398804480, 'GPN': 56425021440, 'PGR': 55696228352, 'ICE': 55628853248, 'ADSK': 54685990912, 'BSX': 54112260096, 'DG': 53191172096, 'NEM': 52692250624, 'PNC': 52284702720, 'NOC': 51937034240, 'MCO': 51664719872, 'WM': 51414249472, 'EW': 50938060800, 'HCA': 50478780416, 'ADI': 50276790272, 'KMB': 47277338624, 'AON': 47212584960, 'ETN': 45563965440, 'EMR': 45207834624, 'DD': 45161127936, 'ILMN': 44932960256, 'AEP': 44568379392, 'LVS': 44280176640, 'MNST': 44191932416, 'GD': 43325296640, 'BAX': 41989234688, 'EXC': 41793503232, 'ROP': 41483169792, 'CTSH': 41402601472, 'LHX': 40870985728, 'PSA': 40641744896, 'MET': 40291614720, 'DLR': 40032776192, 'CNC': 39950983168, 'COF': 39428763648, 'STZ': 39006142464, 'XEL': 38954827776, 'MAR': 38569754624, 'KHC': 38561435648, 'SRE': 38507606016, 'IDXX': 38403723264, 'DOW': 38304071680, 'ROST': 38192324608, 'BIIB': 37965766656, 'COP': 37335879680, 'CTAS': 37037314048, 'GIS': 37022388224, 'WBA': 36849360896, 'APH': 36805042176, 'KLAC': 36528246784, 'SYY': 36503212032, 'INFO': 36298473472, 'ALGN': 36113481728, 'TEL': 36027834368, 'TT': 35466461184, 'EA': 35113820160, 'CMG': 34965864448, 'CMI': 34805329920, 'BK': 34337769472, 'TWTR': 34263681024, 'A': 34231660544, 'SNPS': 34131273728, 'TRV': 34090323968, 'F': 33803925504, 'PPG': 33654345728, 'SBAC': 33623818240, 'IQV': 33345945600, 'ORLY': 33291487232, 'PH': 33287071744, 'EBAY': 33067661312, 'VRSK': 32929351680, 'PAYX': 32854704128, 'MCHP': 32789149696, 'DXCM': 32483254272, 'WEC': 32426719232, 'JCI': 32388517888, 'CDNS': 32160745472, 'MSCI': 32142950400, 'RSG': 32057360384, 'AIG': 32001384448, 'ES': 31981205504, 'HSY': 31775264768, 'TROW': 31560230912, 'ZBH': 31336286208, 'PCAR': 31115509760, 'XLNX': 31059408896, 'YUM': 30930020352, 'RMD': 30830667776, 'BLL': 30830305280, 'VFC': 30638712832, 'TDG': 30250909696, 'PEG': 30108073984, 'KMI': 29780158464, 'APTV': 29756753920, 'BBY': 29483554816, 'MCK': 29419339776, 'ALL': 29389692928, 'AWK': 29244612608, 'AFL': 29228736512, 'HLT': 29142401024, 'SWK': 29106911232, 'FCX': 29006551040, 'PRU': 28985219072, 'MSI': 28779921408, 'OTIS': 28378605568, 'MTD': 28070488064, 'ALXN': 28027480064, 'ANSS': 28014962688, 'HRL': 27878848512, 'ROK': 27778400256, 'CPRT': 27718506496, 'ADM': 27571857408, 'WELL': 27375206400, 'GLW': 27366838272, 'FAST': 27284738048, 'ED': 27254173696, 'AME': 26863538176, 'HPQ': 26794377216, 'DHI': 26750287872, 'CTVA': 26716274688, 'AZO': 26699450368, 'WLTW': 26618841088, 'CLX': 26024284160, 'PSX': 25906608128, 'LUV': 25890377728, 'DTE': 25383458816, 'LYB': 25381183488, 'SLB': 25007818752, 'KR': 24774529024, 'EOG': 24773146624, 'EIX': 24438755328, 'MKC': 24435255296, 'WMB': 24378474496, 'MPC': 24165177344, 'STT': 24110215168, 'FTV': 23854931968, 'ODFL': 23829925888, 'SWKS': 23593834496, 'AVB': 23512248320, 'DFS': 23376449536, 'DAL': 23137562624, 'LEN': 23107006464, 'FRC': 23050174464, 'PPL': 23030212608, 'CERN': 23003449344, 'TSN': 22609022976, 'VRSN': 22601768960, 'SPG': 22568622080, 'K': 22434146304, 'DLTR': 22325936128, 'EQR': 22257006592, 'ABC': 22149406720, 'AJG': 22145878016, 'GRMN': 22059186176, 'ARE': 22047152128, 'O': 22010583040, 'ETR': 21918394368, 'PAYC': 21736685568, 'AMP': 21733906432, 'GWW': 21712463872, 'WY': 21661194240, 'MXIM': 21640687616, 'CHD': 21627574272, 'KEYS': 21592098816, 'FLT': 21492592640, 'WST': 21405251584, 'VLO': 20881854464, 'NDAQ': 20711290880, 'ANET': 20522926080, 'AEE': 20228950016, 'MKTX': 19869337600, 'EFX': 19819292672, 'CDW': 19646550016, 'LH': 19510194176, 'CBRE': 19342393344, 'NTRS': 18841192448, 'FTNT': 18806161408, 'IP': 18793873408, 'VIAC': 18664931328, 'TTWO': 18657767424, 'VTR': 18627416064, 'CMS': 18601689088, 'AMCR': 18484537344, 'ROL': 18460456960, 'BKR': 18344806400, 'ZBRA': 18310983680, 'VMC': 18270617600, 'FITB': 18198767616, 'COO': 18181875712, 'BIO': 18089633792, 'INCY': 18037604352, 'HOLX': 18035714048, 'IR': 17944590336, 'SIVB': 17853614080, 'EXPE': 17618259968, 'DOV': 17592094720, 'SYF': 17537112064, 'KSU': 17535324160, 'CAG': 17396127744, 'CTLT': 17102239744, 'XYL': 17043639296, 'TFX': 16918614016, 'TER': 16901179392, 'BR': 16853203968, 'QRVO': 16792169472, 'CAH': 16733798400, 'DISH': 16673123328, 'TYL': 16642912256, 'ESS': 16515673088, 'AKAM': 16451147776, 'STE': 16389033984, 'MLM': 16355117056, 'DGX': 16332278784, 'HIG': 16157099008, 'FE': 16001785856, 'PEAK': 15992013824, 'TIF': 15952741376, 'RCL': 15887617024, 'FOXA': 15857330176, 'NVR': 15837238272, 'VAR': 15835610112, 'FOX': 15736543232, 'ETSY': 15623632896, 'NUE': 15586839552, 'KMX': 15529194496, 'MTB': 15348246528, 'DPZ': 15326561280, 'TSCO': 15297220608, 'EXPD': 15215560704, 'EXR': 15066216448, 'URI': 15065310208, 'MAA': 15039654912, 'CE': 14927491072, 'DRE': 14879900672, 'CPB': 14835461120, 'PXD': 14821292032, 'ULTA': 14787066880, 'PKI': 14780567552, 'IEX': 14571851776, 'KEY': 14515271680, 'CTXS': 14493541376, 'MAS': 14380404736, 'WAT': 14362491904, 'RF': 14344231936, 'CCL': 14257659904, 'GPC': 14234208256, 'BXP': 14199305216, 'OKE': 14158488576, 'IT': 14114472960, 'FMC': 14065658880, 'LNT': 14049964032, 'STX': 14041333760, 'LYV': 13970276352, 'LDOS': 13968805888, 'DRI': 13963950080, 'AES': 13924516864, 'TDY': 13891307520, 'POOL': 13722996736, 'JBHT': 13671691264, 'CFG': 13644982272, 'SJM': 13491531776, 'J': 13389288448, 'HES': 13345565696, 'WAB': 13254302720, 'CNP': 13072979968, 'ALB': 13047369728, 'HPE': 12992439296, 'EVRG': 12853435392, 'PFG': 12706169856, 'EMN': 12650001408, 'ATO': 12636486656, 'HAL': 12622470144, 'CINF': 12590189568, 'MGM': 12563338240, 'CHRW': 12539658240, 'WDC': 12528808960, 'AVY': 12484292608, 'OMC': 12470583296, 'JKHY': 12392224768, 'PKG': 12329901056, 'DVA': 12328960000, 'ABMD': 12302497792, 'WRB': 12286555136, 'IFF': 12049210368, 'HAS': 12019658752, 'HBAN': 11902527488, 'WHR': 11809479680, 'NLOK': 11721394176, 'FBHS': 11663088640, 'RJF': 11633539072, 'L': 11528172544, 'PHM': 11482492928, 'UDR': 11411176448, 'UHS': 11350539264, 'XRAY': 11303665664, 'UAL': 11291963392, 'OXY': 11020858368, 'NTAP': 10978147328, 'LKQ': 10911041536, 'LUMN': 10773816320, 'DISCK': 10742935552, 'DISCA': 10711126016, 'IPGP': 10703847424, 'WRK': 10538625024, 'ALLE': 10516376576, 'LW': 10482263040, 'AAP': 10384791552, 'BEN': 10110120960, 'PNW': 10085313536, 'NWSA': 10010193920, 'NWS': 9994976256, 'CXO': 9936958464, 'WYNN': 9916580864, 'TXT': 9905710080, 'HWM': 9734410240, 'FFIV': 9718712320, 'GL': 9626811392, 'LB': 9555212288, 'TAP': 9553792000, 'PWR': 9505034240, 'CBOE': 9449353216, 'HSIC': 9401799680, 'HST': 9349148672, 'RE': 9337585664, 'NI': 9266066432, 'SNA': 9250095104, 'BWA': 9213475840, 'AOS': 8874911744, 'MHK': 8782347264, 'PNR': 8753126400, 'WU': 8744543232, 'MYL': 8434578432, 'NWL': 8369317888, 'IPG': 8268742656, 'GPS': 8243329536, 'LNC': 8035958784, 'AIZ': 7823199744, 'NRG': 7822397952, 'REG': 7711955968, 'IRM': 7292167680, 'TPR': 7270784512, 'VNO': 7254529536, 'RHI': 7156804096, 'JNPR': 7156161536, 'IVZ': 7106328576, 'COG': 7038922752, 'CMA': 7028116480, 'AAL': 6868690944, 'MOS': 6789501440, 'PRGO': 6685793280, 'SEE': 6676233216, 'HII': 6631068160, 'FRT': 6584324096, 'CF': 6567221248, 'UA': 6342460416, 'UAA': 6321796608, 'ZION': 6166031360, 'KIM': 6139323392, 'LEG': 5587000832, 'RL': 5574896128, 'NLSN': 5542409216, 'NCLH': 5536148992, 'ALK': 5530872320, 'FANG': 5349755392, 'PBCT': 5346951168, 'DXC': 5245975552, 'PVH': 4945228800, 'FLIR': 4862200832, 'VNT': 4798801920, 'AIV': 4640153088, 'HBI': 4500345856, 'DVN': 4490549760, 'XRX': 4255358208, 'UNM': 4240242688, 'FLS': 4239887104, 'NOV': 4220429568, 'MRO': 3943007744, 'SLG': 3932183808, 'APA': 3869225216, 'HFC': 3590296576, 'FTI': 3315097344}

def get_sp500_sectors():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    # tickers = []
    sectors = {}
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip('\n')
        sector = row.findAll('td')[3].text.strip('\n')
        sectors[ticker] = sector
        # tickers.append(ticker.strip('\n'))
    return sectors

def get_sp500_names():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    # tickers = []
    names = {}
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip('\n')
        name = row.findAll('td')[1].text.strip('\n')
        names[ticker] = name
        # tickers.append(ticker.strip('\n'))
    return names

sectors = get_sp500_sectors()
stock_names = get_sp500_names()
print(sectors)

stock_ohlc_df = pd.read_csv('./stock_data1.csv')
stock_ohlc_df = stock_ohlc_df.set_index('Date')



ohlc_df = stock_ohlc_df
ohlc_df.index = pd.to_datetime(ohlc_df.index)
data = ohlc_df.pct_change()

data = data.fillna(0)
data.index = pd.to_datetime(data.index)
init_names = data.columns
init_sizes = [transform_mc(market_caps[n]) for n in init_names]
init_offsets = [s*750 for s in init_sizes]
init_sectors = [sectors[n] if n in sectors else 'Misc' for n in init_names]
init_full_names = [stock_names[n] if n in stock_names else 'NA' for n in init_names]
init_coords = recursive_mds(end_date='2015-05-02')
init_x, init_y = list(init_coords[:, 0]), list(init_coords[:, 1])
pal_len = len(pd.Series(init_sectors).unique())
new_pal = turbo(pal_len)
init_colors=factor_cmap('sector', new_pal, pd.Series(init_sectors).unique(), nan_color='black') 


ds = ColumnDataSource(dict(x=init_x, y=init_y, name=init_names, full_name=init_full_names, sector=init_sectors, size=init_sizes, label_offset=init_offsets))


plot = figure(plot_width=1100, plot_height=800, x_range=(-1.0,1.0), y_range=(-1.0,1.0),
              toolbar_location="below", background_fill_color="#fafafa")
plot.axis.visible = False
plot.xgrid.visible = False
plot.ygrid.visible = False

labels = LabelSet(x='x', y='y', text='name', level='glyph',
              x_offset=0, y_offset='label_offset', source=ds, render_mode='canvas', text_color='grey')
plot.add_layout(labels)





af_button = Button(label="animate forward >>", button_type="success")
ab_button = Button(label="<< animate backward", button_type="default")

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

date_picker = DatePicker(value="2018-01-01", min_date="2016-01-01", max_date="2020-01-01")

as_slider = Slider(start=1, end=10, value=5, step=.1, title="animation speed")
l_slider = RangeSlider(start=-1.00, end=1.00, value=(0.75, 1.0), step=0.01, title="show links for")
spacer = Spacer(width=240, height=500)
controls = [spacer, date_picker, af_button, ab_button, as_slider, l_slider, checkbox_button_group]



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
    legend_group='sector'
)
plot.legend.background_fill_alpha = 0.5
plot.legend.location = "bottom_right"
main_plot = row(plot, width=1200, height=810)
main_plot.sizing_mode = "fixed"
inputs = column(*controls, width=240, height=800)
inputs.sizing_mode = "fixed"
desc = Div(text=open("description.html").read(), sizing_mode="stretch_width")

TOOLTIPS = [
    ("ticker", "@name"),
    ("name", "@full_name"),
    ("Sector", "@sector")


]
plot.add_tools(HoverTool(tooltips=TOOLTIPS, renderers=[plot_scatter]))

l = layout([

    [inputs, main_plot]
    
], sizing_mode="scale_both")

doc.add_root(l)
