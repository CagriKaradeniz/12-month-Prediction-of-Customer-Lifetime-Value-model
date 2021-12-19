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

# credentials.database: group_5
# user: group_5
# password: miuul
# host: 34.79.73.237
# port: 3306

creds = {'user': 'group_5',
         'passwd': 'miuul',
         'host': '34.79.73.237',
         'port': 3306,
         'db': 'group_5'}

# MySQL conection string.
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection.
conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)



pd.read_sql_query("select * from online_retail_2010_2011 limit 10", conn)
retail_mysql_df = pd.read_sql_query("select * from online_retail_2010_2011", conn)


retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()
data = retail_mysql_df.copy()
data=data[data["Country"]=="United Kingdom"]

data.describe().T
data.dropna(inplace=True)
data = data[~data["Invoice"].str.contains("C", na=False)]
data = data[data["Quantity"] > 0]
data = data[data["Price"] > 0]

util.replace_with_thresholds(data,"Quantity")
util.replace_with_thresholds(data,"Price")
data.describe().T

data["TotalPrice"]=data["Quantity"]*data["Price"]

today_date = dt.datetime(2011, 12, 11)

cltv_data = data.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                         lambda date: (today_date - date.min()).days],
                                         'Invoice': lambda num: num.nunique(),
                                         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_data.columns.droplevel(0)
cltv_data.columns = ['recency', 'T', 'frequency', 'monetary']

cltv_data["monetary"] = cltv_data["monetary"] / cltv_data["frequency"]

cltv_data = cltv_data[(cltv_data['frequency'] > 1)]

cltv_data.shape
cltv_data = cltv_data[cltv_data["monetary"] > 0]
cltv_data.shape

cltv_data["recency"] = cltv_data["recency"] / 30

cltv_data["T"] = cltv_data["T"] / 30

cltv_data["frequency"] = cltv_data["frequency"].astype(int)

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_data['frequency'],cltv_data['recency'],cltv_data['T'])
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_data['frequency'], cltv_data['monetary'])
#GÖREV1
#2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.
#Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary'],
                                   time=6,  # 6 aylık
                                   freq="M",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
#6Aylık kazanç tahmini
cltv=cltv.reset_index()

cltv=cltv.sort_values("clv",ascending=False)
cltv.describe().T
cltv.head()

#GÖREV2
#Farklı zaman periyotlarından oluşan CLTV analizi

cltv1ay=ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary'],
                                   time=1,  # 1 aylık
                                   freq="M",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv12ay=ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary'],
                                   time=12,  # 12 aylık
                                   freq="M",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv1ay=cltv1ay.reset_index()
cltv12ay=cltv12ay.reset_index()
cltv1ay=cltv1ay.sort_values("clv",ascending=False)
cltv12ay=cltv12ay.sort_values("clv",ascending=False)

cltv1ay.CustomerID.head(10),cltv12ay.head(10)
#İlk 10 kontrol ediliğinde 12 aylık periyotta da 1 aylık periyottada ilk 5 sabit kalmış. diğer 5 kişilik grubun sıralamasında
#kısmi değişiklikler var. Bunun sebebi müşterilerin alışveriş frekansı ile ilgili olabilir. 1 aylık periyotta freknası denk gelmemiş
#müşteriler olabilir. İki tablo arasındaki farkın temel sebebi budur.

#GÖREV3
#Segmentasyon ve Aksiyon Önerileri

cltv6ay=ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary'],
                                   time=6,  # 6 aylık
                                   freq="M",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)
cltv6ay=cltv6ay.reset_index()
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv6ay[["clv"]])
cltv6ay["scaled_clv"] = scaler.transform(cltv6ay[["clv"]])
cltv6ay["segment"] = pd.qcut(cltv6ay["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv6ay.groupby("segment").agg({"count", "mean", "sum"})

#6 aylık tahminlemelere bakıldığında neredeyse total kazancın %65'e yakınının A segmenti kişilerden sağlandığı görülmektedir.
#A segmenti ortalama aynı ortalamadadır. A segmenti müşterilere premium hesaplar açılarak önceliklendirmeler ve kampanyalar sağlanabilir.
#Bu yolla A segmentinin tüketim freknasını artırarak ortalama gelirini 0.04 scaled seviyesinden 0.05 seviyesine çekmek hedeflenebilir.
#B segment müşterilerin ürün tercihleri öğrenilerek ürün yelpazesi genişletilebilir.
#B segment ortalama gelirini A seviyesine bu şekilde yaklaştırabliriz


#GÖREV4
#Veri tabanına kayıt gönderme

#Customer ID | recency | T | frequency | monetary | expected_purc_1_week |
#expected_purc_1_month | expected_average_profit clv | scaled_clv | segment

cltv_data=cltv_data.reset_index()
cltv_data["expected_average_profit clv"] = ggf.conditional_expected_average_profit(cltv_data['frequency'],
                                                                                 cltv_data['monetary'])
cltv_data["expected_purc_1_week"] = bgf.predict(1*7/30,
                                                  cltv_data['frequency'],
                                                  cltv_data['recency'],
                                                  cltv_data['T'])
cltv_data["expected_purc_1_month"] = bgf.predict(1,
                                                  cltv_data['frequency'],
                                                  cltv_data['recency'],
                                                  cltv_data['T'])

cltv_data=cltv_data.merge(cltv6ay,on="CustomerID")
cltv_data.drop(columns="clv",inplace=True)

cltv_data=cltv_data[['CustomerID', 'recency', 'T', 'frequency', 'monetary',
        'expected_purc_1_week',
       'expected_purc_1_month','expected_average_profit clv', 'scaled_clv', 'segment']]

cltv_data["CustomerID"] = cltv_data["CustomerID"].astype(int)

cltv_data.to_sql(name='cagri_karadeniz', con=conn, if_exists='replace', index=False)



pd.read_sql_query("select * from cagri_karadeniz limit 10", conn)