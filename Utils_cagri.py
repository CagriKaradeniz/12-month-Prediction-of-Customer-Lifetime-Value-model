import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def dataset_yukle(dataset):
    return pd.read_csv(dataset+".csv")

def degisken_tiplerine_ayirma(data,cat_th,car_th):
   """
   Veri:data parametresi ili fonksiyona girilen verinin değişkenlerin sınıflandırılması.
   Parameters
   ----------
   data: pandas.DataFrame
   İşlem yapılacak veri seti

   cat_th:int
   categoric değişken threshold değeri

   car_th:int
   Cardinal değişkenler için threshold değeri

   Returns
   -------
    cat_deg:list
    categorik değişken listesi
    num_deg:list
    numeric değişken listesi
    car_deg:list
    categoric ama cardinal değişken listesi

   Examples
   -------
    df = dataset_yukle("breast_cancer")
    cat,num,car=degisken_tiplerine_ayirma(df,10,20)
   Notes
   -------
    cat_deg + num_deg + car_deg = toplam değişken sayısı

   """


   num_but_cat=[i for i in data.columns if data[i].dtypes !="O" and data[i].nunique() < cat_th]

   car_deg=[i for i in data.columns if data[i].dtypes == "O" and data[i].nunique() > car_th]

   num_deg=[i for i in data.columns if data[i].dtypes !="O" and i not in num_but_cat]

   cat_deg = [i for i in data.columns if data[i].dtypes == "O" and i not in car_deg]

   cat_deg = cat_deg+num_but_cat

   print(f"Dataset kolon/değişken sayısı: {data.shape[1]}")
   print(f"Dataset satır/veri sayısı: {data.shape[0]}")
   print("********************************************")
   print(f"Datasetin numeric değişken sayısı: {len(num_deg)}")
   print(f"Datasetin numeric değişkenler: {num_deg}")
   print("********************************************")
   print(f"Datasetin categoric değişken sayısı: {len(cat_deg)}")
   print(f"Datasetin categoric değişkenler: {cat_deg}")
   print("********************************************")
   print(f"Datasetin cardinal değişken sayısı: {len(car_deg)}")
   print(f"Datasetin cardinal değişkenler: {car_deg}")
   print("********************************************")

   return cat_deg,num_deg,car_deg

def categoric_ozet(data,degisken,plot=False,null_control=False):
    """
    Task
    ----------
    Datasetinde bulunan categoric değişkenlerin değişken tiplerinin sayısını ve totale karşı oranını bulur.
    Ayrıca isteğe bağlı olarak değişken dağılımının grafiğini ve değişken içinde bulunan null sayısını çıkartır.

    Parameters
    ----------
    data:pandas.DataFrame
    categoric değişkenin bulunduğu dataset.
    degisken:String
    Categoric değişken ismi.
    plot:bool
    Fonksiyonda categoric değişken dağılımının grafiğini çizdirmek için opsiyonel özellik.
    null_control:bool
    Fonksiyonda değişken içinde null değer kontolü için opsiyonel özellik

    Returns
    -------
    tablo:pandas.DataFrame
    Unique değişkenlerin ratio olarak oran tablosu
    Examples
    -------
    df=dataset_yukle("titanic")
    cat_deg,num_deg,car_deg=degisken_tiplerine_ayirma(df,10,20)
    for i in cat_deg:
        tablo=categoric_ozet(df,i,True,True)
    """

    print(pd.DataFrame({degisken: data[degisken].value_counts(),
                        "Ratio": 100 * data[degisken].value_counts() / len(data)}))
    tablo=pd.DataFrame({degisken: data[degisken].value_counts(),
                        "Ratio": 100 * data[degisken].value_counts() / len(data)})
    print("##########################################")
    if plot:
        sns.countplot(x=data[degisken], data=data)
        plt.show()
    if null_control:
        print(f"Null veri sayısı: {data[degisken].isnull().sum()}")

    return tablo
def dataset_ozet(data, head=5):
    print("##################### Shape #####################")
    print(f"Satır sayısı: {data.shape[0]}")
    print(f"Kolon sayısı: {data.shape[1]}")

    print("##################### Types #####################")
    print(data.dtypes)

    print("##################### Head #####################")
    print(data.head(head))

    print("##################### Tail #####################")
    print(data.tail(head))

    print("##################### NA Kontrolü #####################")
    print(data.isnull().sum())

    print("##################### Quantiles #####################")
    print(data.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    print("##################### Describe Tablosu #####################")
    print(data.describe().T)

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit