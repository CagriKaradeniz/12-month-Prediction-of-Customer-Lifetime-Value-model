import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
import Utils_cagri as util
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
#Loading datasets
ddata = pd.read_excel("online_retail_II2010-2011.xlsx")
data=data[data["Country"]=="United Kingdom"]

#Preprocessing Dataset
data.describe().T
data.dropna(inplace=True)
data = data[~data["Invoice"].str.contains("C", na=False)]
data = data[data["Quantity"] > 0]
data = data[data["Price"] > 0]
#First Analysis
util.replace_with_thresholds(data,"Quantity")
util.replace_with_thresholds(data,"Price")
data.describe().T

#New parameters
data["TotalPrice"]=data["Quantity"]*data["Price"]

today_date = dt.datetime(2011, 12, 11)
#Creating CLTV dataset
cltv_data = data.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_data.columns.droplevel(0)
cltv_data.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_data["monetary"] = cltv_data["monetary"] / cltv_data["frequency"]
cltv_data = cltv_data[(cltv_data['frequency'] > 1)]
#First Analysis of CLTV Dataset
cltv_data.shape
cltv_data = cltv_data[cltv_data["monetary"] > 0]
cltv_data.shape
#Set time parameters to month
cltv_data["recency"] = cltv_data["recency"] / 30
cltv_data["T"] = cltv_data["T"] / 30

cltv_data["frequency"] = cltv_data["frequency"].astype(int)
#Loading Beta and Gamma models and Fİt them
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_data['frequency'],cltv_data['recency'],cltv_data['T'])
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_data['frequency'], cltv_data['monetary'])

#Prediction of 12 monthos sales according to cltv

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary'],
                                   time=12,  # 12 aylık
                                   freq="M",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv=cltv.reset_index()
cltv=cltv.sort_values("clv",ascending=False)
cltv.describe().T
cltv.head()





#Split segmentation based on clv values

cltv12ay=ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary'],
                                   time=12,  # 12 aylık
                                   freq="M",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv12ay=cltv12ay.reset_index()
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv6ay[["clv"]])
cltv12ay["scaled_clv"] = scaler.transform(cltv12ay[["clv"]])
cltv12ay["segment"] = pd.qcut(cltv12ay["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv12ay.groupby("segment").agg({"count", "mean", "sum"})



#FINAL
#Add 12 month expected purch data to main dataset


cltv_data["expected_purc_12_month"] = bgf.predict(1*12,
                                                  cltv_data['frequency'],
                                                  cltv_data['recency'],
                                                  cltv_data['T'])





