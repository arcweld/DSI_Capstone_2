from pyspark.sql import SparkSession


if __name__ == '__main__':
    spark = SparkSession.builder \
        .master("local") \
        .appName("Word Count") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    print(spark.range(5000).where("id > 500").selectExpr("sum(id)").collect())


jupyter toree install --spark_home=/home/arcweld/anaconda3/lib/python3.7/site-packages/pyspark --interpreters=Scala,PySpark
