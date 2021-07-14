import os
import zipfile
import urllib
from pyspark.sql import SparkSession
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


DOWNLOAD_ROOT = 'https://ti.arc.nasa.gov/'
ENG_PATH = os.path.join('data','nasa_eng')
ENG_URL = DOWNLOAD_ROOT + 'c/6' 

def fetch_eng_data(eng_url=ENG_URL, eng_path=ENG_PATH):
    os.makedirs(eng_path, exist_ok=True)
    if not os.path.exists(os.path.join(eng_path,'CMAPSSData.zip')):
        zip_path = os.path.join(eng_path, 'CMAPSSData.zip')
        urllib.request.urlretrieve(eng_url, zip_path)
        eng_zip = zipfile.ZipFile(zip_path, 'r')
        eng_zip.extractall(path=eng_path)
        eng_zip.close()
        print('All files downloaded and extracted')
    else:
        print('All files in place')
    
def eng_spark():
    spark = SparkSession.builder \
        .master("local") \
        .appName("phm") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    
    
def load_train_data(series_n=1):
    new_col = ["id","cycle","setting1","setting2","setting3","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21", 'x1', 'x2']
    df = pd.read_csv(f'data/nasa_eng/train_FD00{series_n}.txt', sep=' ', names=new_col)
    df.drop(['x1', 'x2'], axis=1, inplace=True)
    return df

    
def load_test_data(series_n=1):
    new_col = ["id","cycle","setting1","setting2","setting3","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21", 'x1', 'x2']
    df = pd.read_csv(f'data/nasa_eng/test_FD00{series_n}.txt', sep=' ', names=new_col, header=None)
    df.drop(['x1', 'x2'], axis=1, inplace=True)
    truth = pd.read_csv(f'data/nasa_eng/RUL_FD00{series_n}.txt', header=None, names=['eol'])
    truth['id'] = truth.index +1
    df = df.merge(truth, on=['id'], how='left')
    return df

def normalize_data(df1, *df2):
    scale = MinMaxScaler()
    scale.fit(df1.loc[:,'s1':'s21'])
    scale.transform(df1.loc[:,'s1':'s21'])
    if df2:
        scale.transform(df2.loc[:,'s1':'s21'])
        return df1, df2
    return df1

def add_labels(df):
    if 'eol' not in df.columns:
        fail = df.groupby('id').cycle.max()
        df['eol'] = df['id'].apply(lambda x: fail[x])

    df['RUL'] = df['eol'] - df['cycle']
    df.drop('eol', axis=1, inplace=True)
    df['f30'] = df['RUL'].apply(lambda x: 2 if x <= 10 else 1 if x <=30 else 0)
    df['f10'] = df['RUL'].apply(lambda x: 1 if x <= 10 else 0)

    return df

def window_avg_sd(df, win=5):
    win = 5
    sensor = {f's{i}': [f'a{i}', f'sd{i}'] for i in range(1,22) }
    if 'a1' in df.columns: 
        print('DF previously transformed')
        return None
    for s, agg in sensor.items():
        for id in df.id.unique():
            avg = df[s].groupby(df['id']).rolling(window=win, min_periods=4).mean()
            sd = df[s].groupby(df['id']).rolling(window=win, min_periods=4).std()
        df.insert(len(df.columns)-4, f'{agg[0]}', avg.values)
        df.insert(len(df.columns)-4, f'{agg[1]}', sd.values)
    
    return df

# Tensorflow functions

# function to reshape features into (samples, time steps, features) 
def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]
        
# function to generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]

