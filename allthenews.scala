// Databricks notebook source
// MAGIC %md
// MAGIC # Project all the news sentiments 
// MAGIC 
// MAGIC ### clustering Kmeans

// COMMAND ----------

// MAGIC %python
// MAGIC %AddJar http://cedric.cnam.fr/~ferecatu/RCP216/tp/tptexte/lsa.jar

// COMMAND ----------

// MAGIC %python
// MAGIC import org.apache.spark.SparkContext
// MAGIC import org.apache.spark.SparkContext._
// MAGIC import org.apache.spark.SparkConf
// MAGIC import org.apache.spark.rdd.RDD
// MAGIC import org.apache.spark.rdd
// MAGIC 
// MAGIC import scala.xml._
// MAGIC 
// MAGIC import org.apache.hadoop.io.{ Text, LongWritable }
// MAGIC import org.apache.hadoop.conf.Configuration
// MAGIC 
// MAGIC import com.cloudera.datascience.common.XmlInputFormat
// MAGIC import com.cloudera.datascience.lsa.ParseWikipedia._
// MAGIC import com.cloudera.datascience.lsa.RunLSA._
// MAGIC 
// MAGIC import org.apache.spark.mllib.linalg._
// MAGIC import org.apache.spark.mllib.linalg.distributed.RowMatrix
// MAGIC import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector, Vector => BVector}
// MAGIC import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
// MAGIC import org.apache.spark.mllib.feature.StandardScaler
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC import java.io.StringReader
// MAGIC import edu.stanford.nlp.ling.CoreLabel;
// MAGIC import edu.stanford.nlp.ling.HasWord;
// MAGIC import edu.stanford.nlp.process.CoreLabelTokenFactory;
// MAGIC import edu.stanford.nlp.process.DocumentPreprocessor;
// MAGIC import edu.stanford.nlp.process.PTBTokenizer;
// MAGIC 
// MAGIC import java.sql.Timestamp
// MAGIC import java.text.SimpleDateFormat
// MAGIC 
// MAGIC import org.apache.spark.mllib.clustering.{ KMeans, KMeansModel }
// MAGIC import org.apache.spark.mllib.util.KMeansDataGenerator
// MAGIC 
// MAGIC import scala.collection.mutable.HashMap
// MAGIC import org.apache.spark.mllib.linalg.{Vector, Vectors}
// MAGIC import org.apache.spark.rdd.RDD

// COMMAND ----------

// MAGIC %python
// MAGIC def toBreeze(v:Vector) = BVector(v.toArray)
// MAGIC def fromBreeze(bv:BVector[Double]) = Vectors.dense(bv.toArray)
// MAGIC def add(v1:Vector, v2:Vector) = fromBreeze(toBreeze(v1) + toBreeze(v2))
// MAGIC def scalarMultiply(a:Double, v:Vector) = fromBreeze(a * toBreeze(v))
// MAGIC 
// MAGIC def stackVectors(v1:Vector, v2:Vector) = {
// MAGIC var v3 = Vectors.zeros(v1.size+v2.size)
// MAGIC     for (i <- 0 until v1.size) {
// MAGIC       BVector(v3.toArray)(i) = v1(i);
// MAGIC     }
// MAGIC     for (i <- 0 until v2.size) {
// MAGIC       BVector(v3.toArray)(v1.size+i) = v2(i);
// MAGIC     }
// MAGIC }
// MAGIC 
// MAGIC // SÃ©paration du fichier XML en un RDD oÃ¹ chaque Ã©lÃ©ment est un article
// MAGIC // Retourne un RDD de String Ã  partir du fichier "path"
// MAGIC def loadArticle(sc: SparkContext, path: String): RDD[String] = {
// MAGIC @transient val conf = new Configuration()
// MAGIC conf.set(XmlInputFormat.START_TAG_KEY, "<document>")
// MAGIC conf.set(XmlInputFormat.END_TAG_KEY, "</document>")
// MAGIC val in = sc.newAPIHadoopFile(path, classOf[XmlInputFormat], classOf[LongWritable], classOf[Text], conf)
// MAGIC in.map(line => line._2.toString)
// MAGIC }
// MAGIC 
// MAGIC 
// MAGIC // Pour un élément XML de type "document",
// MAGIC //   - on extrait le champ "date"
// MAGIC //   - on parse la chaÃ®ne de caractÃ¨re au format yyyy-MM-dd HH:mm:ss
// MAGIC //   - on retourne un Timestamp
// MAGIC def extractDate(elem: scala.xml.Elem): java.sql.Timestamp = {
// MAGIC     val dn: scala.xml.NodeSeq = elem \\ "date"
// MAGIC     val x: String = dn.text
// MAGIC     // d'aprÃ¨s l'exemple 2011-05-18 16:30:35
// MAGIC     val format = new SimpleDateFormat("yyyy-MM-dd")
// MAGIC     if (x == "")
// MAGIC       return null
// MAGIC     else {
// MAGIC       val d = format.parse(x.toString());
// MAGIC       val t = new Timestamp(d.getTime());
// MAGIC       return t
// MAGIC     }
// MAGIC }
// MAGIC 
// MAGIC // Pour un élément XML de type "document",
// MAGIC //   - on extrait le champ #field
// MAGIC def extractString(elem: scala.xml.Elem, field: String): String = {
// MAGIC     val dn: scala.xml.NodeSeq = elem \\ field
// MAGIC     val x: String = dn.text
// MAGIC     return x
// MAGIC }
// MAGIC 
// MAGIC def extractInt(elem: scala.xml.Elem, field: String): Int = {
// MAGIC     val dn: scala.xml.NodeSeq = elem \\ field
// MAGIC     val x: Int = dn.text.toInt
// MAGIC     return x
// MAGIC }
// MAGIC 
// MAGIC def extractAll(elem: scala.xml.Elem, whatText: String = "text"): (Int, java.sql.Timestamp, String) = {
// MAGIC     return (extractInt(elem,"docid"), extractDate(elem), extractString(elem,whatText))
// MAGIC }
// MAGIC 
// MAGIC def extractText(elem: scala.xml.Elem): String = {
// MAGIC     return (extractString(elem,"title") + " " + extractString(elem,"text"))
// MAGIC }
// MAGIC 
// MAGIC // NÃ©cessaire, car le type java.sql.Timestamp n'est pas ordonnÃ© par dÃ©faut (étonnant...)
// MAGIC implicit def ordered: Ordering[java.sql.Timestamp] = new Ordering[java.sql.Timestamp] {
// MAGIC def compare(x: java.sql.Timestamp, y: java.sql.Timestamp): Int = x compareTo y
// MAGIC }
// MAGIC 
// MAGIC def hasLetters(str: String): Boolean = {
// MAGIC // While loop for high performance
// MAGIC var i = 0
// MAGIC while (i < str.length) {
// MAGIC   if (Character.isLetter(str.charAt(i))) {
// MAGIC     return true
// MAGIC   }
// MAGIC   i += 1
// MAGIC }
// MAGIC false
// MAGIC }

// COMMAND ----------

// MAGIC %python
// MAGIC     val stopwords = sc.textFile("stopwords.txt").collect.toArray.toSet
// MAGIC     val stopwordsBroadcast = sc.broadcast(stopwords).value

// COMMAND ----------

// MAGIC %python
// MAGIC import org.apache.spark.mllib.feature.Word2VecModel
// MAGIC // lire le Word2VecModel
// MAGIC val w2vModel = Word2VecModel.load(sc, "w2vModel")

// COMMAND ----------

// MAGIC %python
// MAGIC val conf = new SparkConf().setAppName("NEWS").setMaster("local[*]").set("spark.driver.memory", "4g")
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC conf.set("spark.hadoop.validateOutputSpecs", "false")
// MAGIC //conf.set("spark.driver.allowMultipleContexts", "true")
// MAGIC //conf.setMaster("local[*]")
// MAGIC //conf.set("spark.executor.memory", MAX_MEMORY)
// MAGIC //conf.set("spark.driver.memory", MAX_MEMORY)
// MAGIC //conf.set("spark.driver.maxResultSize", MAX_MEMORY)
// MAGIC val sc = new SparkContext(conf)
// MAGIC 
// MAGIC 
// MAGIC val hadoopConf = new org.apache.hadoop.conf.Configuration()

// COMMAND ----------

// MAGIC %python
// MAGIC     var textToExtract = "text";
// MAGIC     var useW2Vec = true;
// MAGIC     var weightTfIdf = true;
// MAGIC     println ("*****");

// COMMAND ----------

// MAGIC %python
// MAGIC     val news_raw = loadArticle(sc, "articles1.xml")/*.sample(false,0.01)*/
// MAGIC     val news_xml: RDD[Elem] = news_raw.map(XML.loadString)

// COMMAND ----------

// MAGIC %python
// MAGIC val articles1 = spark.read.format("csv").option("header","true").option("sep",";").load("articles.csv")

// COMMAND ----------

// MAGIC %python
// MAGIC articles1.show(5)

// COMMAND ----------

// MAGIC %python
// MAGIC val articlesfiltred=articles1.filter("month = 6").select("publication")
// MAGIC val articlesfiltred1=articles1.filter("year = 2016").groupBy("publication").count()
// MAGIC articlesfiltred1.show()
// MAGIC articlesfiltred.count()

// COMMAND ----------

// MAGIC %python
// MAGIC //calcul statistiques
// MAGIC articles1.describe().show()
// MAGIC articles1.count()
// MAGIC schema = StructType([StructField("docid", StringType(), True),StructField("source", StringType(), True),StructField("url", StringType(), True),StructField("title", StringType(), True),StructField("summary", StringType(), True),StructField("text", StringType(), True),StructField("date", StringType(), True)])
// MAGIC 
// MAGIC 
// MAGIC articles1.printSchema

// COMMAND ----------

// MAGIC %python
// MAGIC  val news: RDD[(Int, java.sql.Timestamp, String)] = news_xml.map(e => extractAll(e,textToExtract))
// MAGIC     val newsTitles: RDD[(Int, java.sql.Timestamp, String)] = news_xml.map(e => extractAll(e,"title"))
// MAGIC     val newsSummaries: RDD[(Int, java.sql.Timestamp, String)] = news_xml.map(e => extractAll(e,"text"))

// COMMAND ----------

// MAGIC %python
// MAGIC     val stopwords = sc.textFile("stopwords.txt").collect.toArray.toSet
// MAGIC     val stopwordsBroadcast = sc.broadcast(stopwords).value

// COMMAND ----------

// MAGIC %python
// MAGIC    val lemmatizedWithDate = news.mapPartitions(iter => {
// MAGIC       val pipeline = com.cloudera.datascience.lsa.ParseWikipedia.createNLPPipeline();
// MAGIC       iter.map {
// MAGIC         case (docid, date, text) =>
// MAGIC           (docid.toString, date,
// MAGIC             com.cloudera.datascience.lsa.ParseWikipedia.plainTextToLemmas(text.toLowerCase.split("\\W+").mkString(" "), stopwordsBroadcast, pipeline))
// MAGIC         };
// MAGIC     })

// COMMAND ----------

// MAGIC %python
// MAGIC     val lemmatized = lemmatizedWithDate.map { case (docid, date, text) => (docid, text) }
// MAGIC     val numTerms = 1000;

// COMMAND ----------

// MAGIC %python
// MAGIC     val (termDocMatrix, termIds, docIds, idfs) = com.cloudera.datascience.lsa.ParseWikipedia.termDocumentMatrix(lemmatized, stopwordsBroadcast, numTerms, sc);
// MAGIC     termDocMatrix.cache();

// COMMAND ----------

// MAGIC %python
// MAGIC val termIdsTxT = termIds.map(l => l.toString.filter(c => c != '[' & c != ']'))
// MAGIC 
// MAGIC val outputtermIds= "data/termIds.txt"
// MAGIC sc.makeRDD(termIdsTxT.toList).saveAsTextFile("outputtermIds")

// COMMAND ----------

// MAGIC %python
// MAGIC val mat = new RowMatrix(termDocMatrix)
// MAGIC val k = 10 // nombre de valeurs singuliÃ¨res Ã  garder
// MAGIC val svd = mat.computeSVD(k, computeU=true)
// MAGIC val projections = mat.multiply(svd.V)
// MAGIC val projectionsTxt = projections.rows.map(l => l.toString.filter(c => c != '[' & c != ']'))
// MAGIC // Delete the existing path, ignore any exceptions thrown if the path doesn't exist
// MAGIC val outputProjection = "data/projection_LSA.txt"
// MAGIC 
// MAGIC projectionsTxt.saveAsTextFile(outputProjection)

// COMMAND ----------

// MAGIC %python
// MAGIC         val nbClusters = 10
// MAGIC         val nbIterations = 1000
// MAGIC         val runs = 10
// MAGIC         val clustering = KMeans.train(termDocMatrix, nbClusters, nbIterations, runs, "k-means||", 0)
// MAGIC         /*val outputClustering = "hdfs://head.local:9000/user/emeric/clusters"
// MAGIC         try { hdfs.delete(new org.apache.hadoop.fs.Path(outputClustering), true) } 
// MAGIC         catch { case _ : Throwable => { } }
// MAGIC         clustering.save(sc, outputClustering)*/
// MAGIC         
// MAGIC         val classes = clustering.predict(termDocMatrix)
// MAGIC         val outputClasses = "data/classes_LSA.txt"
// MAGIC     
// MAGIC         classes.saveAsTextFile(outputClasses)

// COMMAND ----------

// MAGIC %python
// MAGIC  
// MAGIC         val outputData = lemmatizedWithDate.zip(classes).map { case ((docid, date, title),cl) => (docid, date, cl) }.sortBy(_._2).map(l => l.toString.filter(c => c != '(' & c != ')'))
// MAGIC         val outputDataFile = "data/output_LSA.txt"
// MAGIC         //try { hdfs.delete(new org.apache.hadoop.fs.Path(outputDataFile), true) } 
// MAGIC         //catch { case _ : Throwable => { } }
// MAGIC         outputData.saveAsSingleTextFile(outputDataFile)

// COMMAND ----------

// MAGIC %python
// MAGIC        
// MAGIC         clustering.clusterCenters.foreach(clusterCenter => {
// MAGIC             val highest = clusterCenter.toArray.zipWithIndex.sortBy(-_._1).map(v => v._2).take(10)
// MAGIC             println("*****")
// MAGIC             highest.foreach { s => print( termIds(s) + "," ) }
// MAGIC             println ()
// MAGIC             }
// MAGIC        )

// COMMAND ----------

// MAGIC %python
// MAGIC import org.apache.spark.mllib.feature.Word2VecModel
// MAGIC // lire le Word2VecModel
// MAGIC val w2vModel = Word2VecModel.load(sc, "w2vModel")

// COMMAND ----------

// MAGIC %python
// MAGIC val vectors = w2vModel.getVectors.mapValues(vv => org.apache.spark.mllib.linalg.Vectors.dense(vv.map(_.toDouble))).map(identity)
// MAGIC         // transmettre la map aux noeuds de calcul
// MAGIC val bVectors = sc.broadcast(vectors)

// COMMAND ----------

// MAGIC %python
// MAGIC 
// MAGIC         val idTerms = termIds.map(_.swap)
// MAGIC 
// MAGIC         val pairs = termDocMatrix.zip(lemmatized);
// MAGIC 
// MAGIC         var w2vecRepr = pairs.map({ case (row, (docid, lemmas)) => 
// MAGIC           var vSum = Vectors.zeros(100)
// MAGIC           var totalWeight: Double = 0
// MAGIC           val words = lemmas.toSet
// MAGIC           words.foreach { word =>
// MAGIC                val colId = idTerms.get(word);
// MAGIC                if (colId != None) {
// MAGIC                    var weight = row(colId.get);
// MAGIC                    if (! weightTfIdf) {
// MAGIC                        weight = 1
// MAGIC                    }
// MAGIC                    val wvec = bVectors.value.get(word)
// MAGIC                    if (wvec != None) {          
// MAGIC                        vSum = add(scalarMultiply(weight, wvec.get), vSum)
// MAGIC                        totalWeight += weight
// MAGIC                    }
// MAGIC                }
// MAGIC           }
// MAGIC           if (totalWeight > 0) {
// MAGIC                vSum = scalarMultiply(1.0 / totalWeight, vSum)
// MAGIC           }
// MAGIC           else {
// MAGIC                vSum = Vectors.zeros(100);
// MAGIC           }
// MAGIC           (docid, vSum)
// MAGIC         })/*.filter(vec => Vectors.norm(vec._2, 1.0) > 0.0)*/.persist() 

// COMMAND ----------

// MAGIC %python
// MAGIC val matRDD = w2vecRepr.map{v => v._2}.cache()
// MAGIC         val mat = new RowMatrix(matRDD)
// MAGIC         val matrixTxt = mat.rows.map(l => l.toString.filter(c => c != '[' & c != ']'))
// MAGIC         // Delete the existing path, ignore any exceptions thrown if the path doesn't exist
// MAGIC         val outputMatrix = "data/matrice_W2V.txt"
// MAGIC         //try { hdfs.delete(new org.apache.hadoop.fs.Path(outputMatrix), true) } 
// MAGIC         //catch { case _ : Throwable => { } }
// MAGIC         matrixTxt.saveAsTextFile(outputMatrix)

// COMMAND ----------

// MAGIC %python
// MAGIC         val centRed = new StandardScaler(withMean = true, withStd = true).fit(matRDD)
// MAGIC         val matCR: RowMatrix = new RowMatrix(centRed.transform(matRDD))
// MAGIC 
// MAGIC         val matCompPrinc = matCR.computePrincipalComponents(10)
// MAGIC         val projections = matCR.multiply(matCompPrinc)
// MAGIC         //val matSummary = projections.computeColumnSummaryStatistics()
// MAGIC         val projectionsTxt = projections.rows.map(l => l.toString.filter(c => c != '[' & c != ']'))
// MAGIC         // Delete the existing path, ignore any exceptions thrown if the path doesn't exist
// MAGIC         val outputProjection = "data/projection_W2V.txt"
// MAGIC         //try { hdfs.delete(new org.apache.hadoop.fs.Path(outputProjection), true) } 
// MAGIC         //catch { case _ : Throwable => { } }
// MAGIC         projectionsTxt.saveAsTextFile(outputProjection)

// COMMAND ----------

// MAGIC %python
// MAGIC      val nbClusters = 10
// MAGIC         val nbIterations = 1000
// MAGIC         val runs = 10
// MAGIC         val clustering = KMeans.train(matRDD, nbClusters, nbIterations, runs, "k-means||", 0)
// MAGIC         /*val outputClustering = "hdfs://head.local:9000/user/emeric/clusters"
// MAGIC         try { hdfs.delete(new org.apache.hadoop.fs.Path(outputClustering), true) } 
// MAGIC         catch { case _ : Throwable => { } }
// MAGIC         clustering.save(sc, outputClustering)*/
// MAGIC         
// MAGIC         val classes = clustering.predict(matRDD)
// MAGIC         val outputClasses = "data/classes_W2V.txt"
// MAGIC         //try { hdfs.delete(new org.apache.hadoop.fs.Path(outputClasses), true) } 
// MAGIC         //catch { case _ : Throwable => { } }
// MAGIC         classes.saveAsTextFile(outputClasses)

// COMMAND ----------

// MAGIC %python
// MAGIC         val outputData = lemmatizedWithDate.zip(classes).map { case ((docid, date, title),cl) => (docid, date, cl) }.sortBy(_._2).map(l => l.toString.filter(c => c != '(' & c != ')'))
// MAGIC         val outputDataFile = "data/output_W2V.txt"
// MAGIC         //try { hdfs.delete(new org.apache.hadoop.fs.Path(outputDataFile), true) } 
// MAGIC         //catch { case _ : Throwable => { } }
// MAGIC         outputData.saveAsTextFile(outputDataFile)

// COMMAND ----------

// MAGIC %python
// MAGIC    clustering.clusterCenters.foreach(clusterCenter => {
// MAGIC             val nearest = w2vecRepr.map{v => (v._1, Vectors.sqdist(v._2,clusterCenter))}.sortBy(_._2).map{ case (id, dist) => id }.take(10).toSet
// MAGIC             println("*****")
// MAGIC             pairs.filter{ case (row, (docid, lemmas)) => nearest contains docid }.foreach {  case (row, (docid, lemmas)) => {
// MAGIC                 val id = row.toArray.zipWithIndex.sortBy(- _._1).take(5).map(_._2)
// MAGIC                 id.foreach { s => print( termIds(s) + "," ) }
// MAGIC                 println ()
// MAGIC                 }
// MAGIC             }

// COMMAND ----------

// MAGIC %python
// MAGIC import org.apache.spark.mllib.feature.Word2VecModel
// MAGIC // lire le Word2VecModel
// MAGIC val w2vModel = Word2VecModel.load(sc, "w2vModelwiki8")
