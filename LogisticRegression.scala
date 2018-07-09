
package example

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.IntegerType


object LogisticRegression {
  def main(args: Array[String]) {
    // create Spark context with Spark configuration

    // reading the data
    val sc = new SparkContext(new SparkConf().setAppName("Logistic Regression").setMaster("local"))
    val sqlContext = new SQLContext(sc)
    val csv = sqlContext.read.option("inferSchema", "true").option("header", "true").csv("/data/flights.csv")
    csv.show()
    val dataold = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")
    // data.schema.add("Late",IntegerType)
    // data.show()
    val data = dataold.withColumn("Late", (dataold.col("DepDelay").>=(15)).cast(IntegerType))
    data.show()
    println(data.columns)
    println(data.schema)
    // .alias("Late"))
    val splits = data.randomSplit(Array(0.7, 0.3))
    val train = splits(0)
    val test = splits(1)
    val train_rows = train.count()
    val test_rows = test.count()
    println("Training Rows: " + train_rows + " Testing Rows: " + test_rows)

    // preparing training  data

    val assembler = new VectorAssembler().setInputCols(Array("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay")).setOutputCol("features")

    // data.col("Late").alias("label")
    val updated_train = train.withColumnRenamed("Late", "label")
    val training = assembler.transform(updated_train).select("features", "label")
    // data.
    training.show()

    //train the model
    val lr = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setMaxIter(10).setRegParam(0.3)
    val model = lr.fit(training)
    println("Model trained!")

    //   test.col("Late").alias("trueLabel")
    val updated_test = test.withColumnRenamed("Late", "trueLabel")
    
    // preparing testing data
    val testing = assembler.transform(updated_test).select("features", "trueLabel")
    testing.show()

    //test the model
    val prediction = model.transform(testing)
    val predicted = prediction.select("features", "prediction", "probability", "trueLabel")
    predicted.show(100)

    // calculating precision and recall

    val tp = predicted.filter("prediction == 1 AND truelabel == 1").count().toFloat
    val fp = predicted.filter("prediction == 1 AND truelabel == 0").count().toFloat
    val tn = predicted.filter("prediction == 0 AND truelabel == 0").count().toFloat
    val fn = predicted.filter("prediction == 0 AND truelabel == 1").count().toFloat
    val metrics = sqlContext.createDataFrame(Seq(
      ("TP", tp),
      ("FP", fp),
      ("TN", tn),
      ("FN", fn),
      ("Precision", tp / (tp + fp)),
      ("Recall", tp / (tp + fn)))).toDF("metric", "value")
    metrics.show()

    prediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate = false)

    // calculating ROC

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("trueLabel").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
    val auc = evaluator.evaluate(prediction)
    
    
    println("AUC = " + (auc))

    // change the dimension threshold

    // Redefine the pipeline
    val lrRedifined = new LogisticRegression().setLabelCol("label").setFeaturesCol("features").setThreshold(0.35).setMaxIter(10).setRegParam(0.3)
    val pipelineRedifined = new Pipeline().setStages(Array(assembler, lrRedifined))

    // Retrain the model
    val modelRedifined = pipelineRedifined.fit(updated_train)

    // Retest the model
    val newPrediction = modelRedifined.transform(updated_test)
    newPrediction.select("rawPrediction", "probability", "prediction", "trueLabel").show(100, truncate = false)

    sqlContext.clearCache()
    // sqlContext.st
    sc.stop()

   

    // Recalculate confusion matrix
    val tp2 = newPrediction.filter("prediction == 1 AND truelabel == 1").count().toFloat
    val fp2 = newPrediction.filter("prediction == 1 AND truelabel == 0").count().toFloat
    val tn2 = newPrediction.filter("prediction == 0 AND truelabel == 0").count().toFloat
    val fn2 = newPrediction.filter("prediction == 0 AND truelabel == 1").count().toFloat
    val metricsNew = sqlContext.createDataFrame(Seq(
      ("TP", tp2),
      ("FP", fp2),
      ("TN", tn2),
      ("FN", fn2),
      ("Precision", tp2 / (tp2 + fp2)),
      ("Recall", tp2 / (tp2 + fn2)))).toDF("metric", "value")
    metricsNew.show()
  }
}