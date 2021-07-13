import os
import zipfile
import urllib
from pyspark.sql import SparkSession


DOWNLOAD_ROOT = 'https://ti.arc.nasa.gov/'
ENG_PATH = os.path.join('data','nasa_eng')
ENG_URL = DOWNLOAD_ROOT + 'c/6' 

def fetch_eng_data(eng_url=ENG_URL, eng_path=ENG_PATH):
    os.makedirs(eng_path, exist_ok=True)
    zip_path = os.path.join(eng_path, 'CMAPSSData.zip')
    urllib.request.urlretrieve(eng_url, zip_path)
    eng_zip = zipfile.ZipFile(zip_path, 'r')
    eng_zip.extractall(path=eng_path)
    eng_zip.close()
    
def eng_spark():
    spark = SparkSession.builder \
        .master("local") \
        .appName("phm") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    