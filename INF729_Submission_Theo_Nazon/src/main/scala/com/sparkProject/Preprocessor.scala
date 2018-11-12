package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions.{concat, concat_ws, when, isnull}
import org.apache.spark.sql.types.{DateType, IntegerType}



object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP
    // on vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation de la SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc et donc aux mécanismes de distribution des calculs.)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/
    val df = spark.read.format("csv").option("header", true).option("inferSchema", true).csv("/home/theo/Documents/MASTER/Theo_Nazon/INF729_Spark/TP_ParisTech_2018_2019_starter/TP_ParisTech_2017_2018_starter/data/train_clean.csv")

    println(s"Total number of rows in the DF is: ${df.count()}")
    println(s"Total number of columns in the DF is: ${df.columns.length}")
    println(s"in the DF is: ${df.columns}")
    df.printSchema()

    val dfCasted = df
      .withColumn("goal", $"goal".cast("int"))
      .withColumn("deadline", $"deadline".cast("int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("int"))
      .withColumn("created_at", $"created_at".cast("int"))
      .withColumn("launched_at", $"launched_at".cast("int"))
      .withColumn("backers_count", $"backers_count".cast("int"))
      .withColumn("final_status", $"final_status".cast("int"))

    println("\n")

    dfCasted.printSchema()

    println("\n")

    dfCasted.select("goal", "backers_count", "final_status").describe().show

    println("\n")
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(50)
    println("\n")
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(50)
    println("\n")
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(50)
    println("\n")
    dfCasted.select("deadline").dropDuplicates.show()
    println("\n")
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(50)
    println("\n")
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(50)
    println("\n")
    dfCasted.select("goal", "final_status").show(30)
    println("\n")
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)
    println("\n")


    val df2 = dfCasted.drop($"disable_communication")

    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    df.filter($"country" === "False").groupBy("currency").count.orderBy($"count".desc).show(50)

    def udfCountry = udf{(country: String, currency: String) =>
      if (country == "False")
        currency
      else
        country
    }

    def udfCurrency = udf{(currency: String) =>
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", udfCountry($"country", $"currency"))
      .withColumn("currency2", udfCurrency($"currency"))
      .drop("country", "currency")

    // Constructing a DataFrame with cleaned view on final_status (removing rows with non 0 or 1 values)

    dfCountry.groupBy("final_status").count.orderBy($"count".desc).show
    println(dfCountry.count())
    println(dfCountry.columns.length)
    val dfCountry2: DataFrame = dfCountry
      .filter($"final_status" === 0 || $"final_status" === 1)

    println("DataFrame dfCountry after removing rows whose final_status is not 0 or 1")
    println(dfCountry2.count())
    println(dfCountry2.columns.length)
    dfCountry2.groupBy("final_status").count.orderBy($"count".desc).show


    // Constructing a DataFrame with cleaned view on final_status (0 replacing non 1 or 0 values)
    def udfFinalStatus = udf{(final_status: Int) =>
      if (final_status == 0 || final_status == 1)
        final_status
      else
        0
    }
    val dfCountry2bis: DataFrame = dfCountry
      .withColumn("final_status", udfFinalStatus($"final_status"))

    println("DataFrame dfCountry after replacing non 0 or 1 values in final_status by 0")
    println(dfCountry2bis.count())
    println(dfCountry2bis.columns.length)
    dfCountry2bis.printSchema()
    dfCountry2bis.groupBy("final_status").count.orderBy($"count".desc).show


    def udfDaysCampaign = udf{(launched_at: Int, deadline: Int) =>
      val valueInSeconds = (deadline - launched_at)/86400
      (valueInSeconds * 10000) / 10000.toDouble
    }

//    def udfMonthLaunch = udf{(launched_at: Int) =>
//      val monthValue = monthValue(launched_at)
//    }


    val dfDate = dfCountry2
      .withColumn("days_campaign", udfDaysCampaign($"launched_at", $"deadline"))

//    val dfDate2 = dfDate
//      .withColumn("monthNumber", udfMonthLaunch($"launched_at"))

//    dfDate.printSchema()
//    dfDate.select("days_campaign", "launched_at", "deadline").show(20)

    def udfHoursPrepa = udf{(created_at: Int, launched_at: Int) =>
      val valueInSeconds = (launched_at - created_at)/3600
      (valueInSeconds * 10000).round / 10000.toDouble
    }


    val dfDate2 = dfDate
      .withColumn("hours_prepa", udfHoursPrepa($"created_at", $"launched_at"))

//    dfDate2.printSchema()
//    dfDate2.select("created_at", "launched_at", "hours_prepa").show(20)

    val dfDataClean = dfDate2
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    dfDataClean.printSchema()
//    dfDataClean.select("text", "name", "desc", "keywords").show(10)


    val dfNull = dfDataClean
      .withColumn("days_campaign", when($"days_campaign".isNull, -1).otherwise($"days_campaign"))
      .withColumn("hours_prepa", when($"hours_prepa".isNull, -1).otherwise($"hours_prepa"))
      .withColumn("goal", when($"goal".isNull, -1).otherwise($"goal"))
      .withColumn("country2", when($"country2".isNull, "unknown").otherwise($"country2"))
      .withColumn("currency2", when($"currency2".isNull, "unknown").otherwise($"currency2"))

    dfNull.printSchema()


    //    dfNull.write.format("parquet").save("finalTab.parquet") //To save the result in parquet format

  }
}
