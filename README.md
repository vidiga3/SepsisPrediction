# BD4H
This repository will capture the work done by the team for spesis prediction using MIMIC III database as part of GeorgiaTech course

## Dependencies <br>
* scalaVersion := "2.11.12" <br>
 

* libraryDependencies ++= Seq(<br>
  "org.apache.spark" %% "spark-core" % "2.3.0", <br>
  "org.apache.spark" %% "spark-sql" % "2.3.1",<br><br>
  "com.databricks" %% "spark-xml" % "0.4.1",<br>
  "com.databricks" %% "spark-csv" % "1.5.0",<br>
  "org.apache.hadoop" % "hadoop-hdfs" % "2.7.2",<br>
  "org.apache.spark" %% "spark-mllib" % "2.3.0"<br>
)

-----

## How to Run the Code <br>
The code is divided into 3 main sections
 * Create features from unstructured data using Physician Notes in Spark
 * Create features using ICD9 Codes in Spark.
 * Create and run models using Python  
 
 ### Spark
  * Execute Spark(main.scala)
  * executes code to idenfity the sepsis date for patients who are identified with angus criteria of sepsis
  * executes code to find the date of final record of patients with non sepsis
  * executes code to filter data to extract information of blood vitals (heart rate, manual BP, Oxygen saturation,blood Temperature
  * extracts 20 topics from the patient notes and saves distribution of those 20 topics for each patient
 
  #### Run Spark code on Docker VM
   You can use the docker VM shared with the Homework assigments to run our Spark Code with sample Data.<br>   
   Connect to docker container and run below command at /source folder<br>
   *sbt compile “run datafolderpath”* where *datafolderpath* is the path where sample data is kept <br>
   
   Note: You will be able to find the sampel data in final assignment upload. Extarct the zip and copy the contents in datafolderpath<br>
 
  #### Run Spark code on Azure Cluster
  
  Generate a JAR package with all dependencies using command *sbt assembly*  
  This will create the jar file in target\scala-2.11  folder.<br>
  
  Upload this JAR to Azure Blob storage as well.  
  Assuming the MIMIC data has been already uploaded to Azure Blob storage.If not please do so and you will find steps in ReadMe.docx submitted with the final project assigment
  
  Login to spark cluster using ssh. You can use *Putty* to login<br>
   Run the command *Spark-submit –class “main class” “path to jar file” “first argument which is full path to data files”*
   e.g. spark-submit --class "edu.gatech.cse6250.main.Main"  "wasb://xyzcontainer@xyzabcstorgaeaccount.blob.core.windows.net/BlobCodeFolderName/big-data-project-1.1.0-2.11.12.jar" "wasb://xyzcontainer@xyzabcstorageaccount.blob.core.windows.net/BlobMIMICDataFolderName/"
   
   *wasb* is used for accessing Azure File System <br>
   You can see the output files under *BlobMIMICDataFolderName*.  We are using the same folder for generating output.
   * Here is the list of output files generated
     * NoSepsisoutputWithSepTime.csv
     * outputWithSepTime.csv
     * filteredChartData.csv
     * topics.csv
       
 
### Python
  * Merge data of sepsis and non sepsis with their max dates
  * Merge date of patients with max dates and their blood vitals information
  * Extract the mean,median,standard deviation,len(count) summary details for each patient for following periods 
    * LastDay -- Day before sepsis was identified
    * last 7 days (week)
    * last 30 days
    * history of patients data
  * code to run logistic regression, CNN, RNN models
  
---

### Final Models
  * Final models are in the following folder: python/finalModels
    
