package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      ********************************************************************************/
    // #############################################################
    // 1. LOADING DATA FRAME
    val parquetFileDF = spark.read.parquet("/home/theo/Documents/MASTER/Theo_Nazon/INF729_Spark/TP_ParisTech_2018_2019_starter/TP_ParisTech_2017_2018_starter/data/prepared_trainingset")
    parquetFileDF.printSchema()

    println(parquetFileDF.head())

    // #############################################################
    // 2. PREPROCESSING TEXT DATA
    // 2.a) Stage 1 of Pipeline
    val tokenizer = new RegexTokenizer().setPattern("\\W+").setGaps(true).setInputCol("text").setOutputCol("tokens")


    // 2.b) Stage 2 of Pipeline
    StopWordsRemover.loadDefaultStopWords("english")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")


    // 2.c) Stage 3 of Pipeline
    val vectorizer = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("vectorized")


    // 2.d) Stage 4 of Pipeline
    val idf = new IDF()
      .setInputCol(vectorizer.getOutputCol)
      .setOutputCol("tfidf")

    // #############################################################
    // 3. Preprocesing - From Text to Numerical Data
    // 3.e) Stage 5 of Pipeline
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")


    // 3.f) Stage 6 of Pipeline
    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    // 3.g) Stage 7/8 of Pipeline
    val encoderCountry = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("countryVect")


    val encoderCurrency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currencyVect")

    // #############################################################
    // 4.h) Stage 9 of Pipeline
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign","hours_prepa", "goal", "country_indexed","currency_indexed"))
      .setOutputCol("features")


    // 4.i) Stage 10 of Pipeline
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    // 4.j CREATING PIPELINE ASSEMBLING STAGE 1 to 10
    val stages = Array(tokenizer, remover, vectorizer, idf, indexerCountry, indexerCurrency, assembler, lr)
    val pipeline = new Pipeline().setStages(stages)

    // #############################################################
    // 5. TRAINING AND TUNING

    val splits = parquetFileDF.randomSplit(Array(0.9, 0.1), seed=1)
    val (training, test) = (splits(0), splits(1))


    // #############################################################
    // 5.l) Preparation of param-grid for Grid Search
    val paramGrid = new ParamGridBuilder()
      .addGrid(vectorizer.minDF, (55.0.to(95.0).by(20.0).toArray))
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)

    val validationModel = trainValidationSplit.fit(training)

    // 5.m) Preparation of param-grid for Grid Search

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.

    val dfPrediction = validationModel
      .transform(test)
      .select("features","final_status", "predictions", "raw_predictions")

    // F1 score calculation
    val metrics = evaluator.evaluate(dfPrediction)
    println("F1 Score du modèle sur le Test set : " + metrics)

    // Display predictions
    dfPrediction.groupBy("final_status","predictions").count.show()
  }
}
