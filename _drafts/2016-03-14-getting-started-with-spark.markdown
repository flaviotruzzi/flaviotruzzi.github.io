---
layout: post
title:  "Getting started with Spark"
date:   2016-03-14 00:58:59
author: Flavio Truzzi
categories: Scala Spark
tags:	scala spark
cover:  https://images.unsplash.com/photo-1439707769435-4bf7b5f265ba?fit=crop&fm=jpg&h=400&q=100&w=1450
---

I am using [Spark](http://spark.apache.org/) for quite some time now. In this post, I will explain how to setup a simple [Spark Self-Contained Application](http://spark.apache.org/docs/latest/quick-start.html#self-contained-applications) in a slightly different way than the one in the quick start documentation, IMHO, easier. At the end of this post, we will have built another word counting application.

# Project Structure

If you want to just get the code just clone the [spark-seed repo](https://github.com/flaviotruzzi/spark-seed).

I will be using sbt, so our directory structure at the end should look like this:

{% highlight bash %}
spark-seed    
├── build.sbt
├── data
│   └── input
│       └── pg1342.txt
└── src
    ├── main
    │   ├── resources
    │   │   └── logback.xml
    │   └── scala
    │       └── Boot.scala
    └── test
        └── scala
            └── TokenizerSpec.scala
{% endhighlight %}

Let's start with the `build.sbt`, if you are not familiar with sbt at all, you should probably start with their [docs](http://www.scala-sbt.org/).
{% highlight bash %}
name := "spark-seed"

version := "0.1"

scalaVersion := "2.11.7"

lazy val sparkVersion = "1.6.0"

libraryDependencies ++= Seq(
  "com.typesafe.scala-logging" %% "scala-logging" % "3.1.0",
  "ch.qos.logback" % "logback-classic" % "1.1.3",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "com.holdenkarau" % "spark-testing-base_2.11" % "1.6.0_0.3.1",
  "org.scalactic" %% "scalactic" % "2.2.6",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test"
)

enablePlugins(JavaAppPackaging)
{% endhighlight %}

We are just defining the name of our project `spark-seed` and our dependencies.
We are using Spark 1.6.0 (`spark-core`), we also added the `spark-testing-base` that contains
have some utilities to facilitate our testing environment.

The last line is enabling the `SBT Native Packager` plugin that we are going to setup right now ;)

On `plugins.sbt` add the following:

{% highlight bash %}
addSbtPlugin("com.typesafe.sbt" % "sbt-native-packager" % "1.0.6")
{% endhighlight %}

That add the `SBT Native Packaging` plugin, that we will use in order to package our application later. On the `build.properties` just drop the following:

{% highlight properties %}
sbt.version=0.13.11
{% endhighlight %}

This will configure sbt to use version `0.13.11`.

Finally the last part of the structure that is not data neither actual code, let's configure our `logback.xml`, that should be put on `src/main/resources`.

{% highlight xml %}
<configuration debug="false">

    <appender name="STDOUT" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%date{HH:mm:ss.SSS} %highlight(%-5level)
                     %gray(%logger{90}) %X{X-ApplicationId} %msg%n
            </pattern>
        </encoder>
    </appender>

    <logger name="httpclient" level="WARN"/>
    <logger name="io.netty" level="WARN"/>
    <logger name="org.apache.commons" level="WARN"/>
    <logger name="org.apache.hadoop" level="WARN"/>
    <logger name="org.apache.spark.deploy.SparkHadoopUtil" level="INFO"/>
    <logger name="org.apache.spark.memory" level="WARN" />
    <logger name="org.apache.spark.scheduler" level="WARN"/>
    <logger name="org.apache.spark.sql.catalyst.expressions.codegen" level="WARN"/>
    <logger name="org.apache.spark.storage" level="WARN"/>
    <logger name="org.apache.spark.ui" level="WARN" />
    <logger name="org.apache.spark.util" level="WARN"/>
    <logger name="org.apache.spark" level="DEBUG"/>
    <logger name="org.apache.spark" level="WARN"/>
    <logger name="org.jets3t" level="WARN"/>
    <logger name="org.spark-project.jetty" level="WARN"/>

</configuration>
{% endhighlight %}

Here we are configuring the format of our logs, and what we really want to log, running Spark on debug mode can be too cluttered.

# Data

Since we decide to build a simple word counting, I took the liberty of choosing the book [Pride and Prejudice, by Jane Austen](https://www.gutenberg.org/ebooks/1342.txt.utf-8), that is freely available through the amazing [Gutenberg project](https://www.gutenberg.org/).

That is not that much data when you consider that we usually want using Spark to analyze huge amounts of data, however, this is just a toy project, a seed for starting your own projects.

The file consists of 701Kbytes, distributed through 13423 lines, and 124588 words.

{% highlight bash %}
➜  spark-seed git:(master) ✗ wc data/input/pg1342.txt
   13426  124588  717574 data/input/pg1342.txt
{% endhighlight %}

# The actual code

Finally, we can start with our code. Let's created the `Boot.scala` file, which will hold an object with our main function. I will start defining the main function, and add things step by step.

First we need to create a [SparkConf](https://spark.apache.org/docs/1.6.0/api/java/org/apache/spark/SparkConf.html) object, this object holds the configuration, and once created it does not support changes, from the documentation:

>"... Note that once a SparkConf object is passed to Spark, it is cloned and can no longer be modified by the user. Spark does not support modifying the configuration at runtime."

{% highlight scala %}
import org.apache.spark.SparkConf

object Boot {

  val numberOfCores = 4

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Spark Seed")
      .setMaster(s"local[$numberOfCores]")
  }
}
{% endhighlight %}

There are some things happening here, we are creating a new instance of the `SparkConf`, setting the name of the app, and setting the master to local with `numberOfCores` threads.

Now we are going to need a [SparkContext](https://spark.apache.org/docs/1.6.0/api/java/org/apache/spark/SparkContext.html), according to the documentation:

> "Main entry point for Spark functionality. A SparkContext represents the connection to a Spark cluster, and can be used to create RDDs, accumulators and broadcast variables on that cluster.
Only one SparkContext may be active per JVM. You must stop() the active SparkContext before creating a new one."

{% highlight scala %}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf

object Boot {

  val numberOfCores = 4

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Spark Seed")
      .setMaster(s"local[$numberOfCores]")

    implicit val sc = new SparkContext(conf)

}
{% endhighlight %}

Now we need to actually get the data, and now it is pretty simple, we just use the `textFile` method of our brand new `SparkContext`:

{% highlight scala %}
    val book = sc.textFile("./data/input/pg1342.txt")
{% endhighlight %}

Note that we could load several books with a wildcard instead of the exact name of the book or using the [`wholeTextFiles`](https://spark.apache.org/docs/1.6.0/api/java/org/apache/spark/SparkContext.html#wholeTextFiles(java.lang.String,%20int)) method.

That method returns an [RDD](https://spark.apache.org/docs/1.6.0/api/scala/index.html#org.apache.spark.rdd.RDD), from the docs:

> "A Resilient Distributed Dataset (RDD), the basic abstraction in Spark. Represents an immutable, partitioned collection of elements that can be operated on in parallel. This class contains the basic operations available on all RDDs, such as map, filter, and persist. In addition, org.apache.spark.rdd.PairRDDFunctions contains operations available only on RDDs of key-value pairs, such as groupByKey and join; org.apache.spark.rdd.DoubleRDDFunctions contains operations available only on RDDs of Doubles; and org.apache.spark.rdd.SequenceFileRDDFunctions contains operations available on RDDs that can be saved as SequenceFiles. All operations are automatically available on any RDD of the right type (e.g. RDD[(Int, Int)] through implicit."

We can think the RDD as a distributed collection, which we can actually do something with it.

With the data in our hands we need to actually process it somehow, for that, let's define a simple tokenizer function, that tokenize by splitting at every space.

{% highlight scala %}
def tokenize(text: RDD[String]): RDD[String] =
  text
    .flatMap(_.split(" "))    
{% endhighlight %}

The code defined on the function `tokenize` is iterating over an `RDD` of strings, and flatMapping splitting on the space. The result of that operation is also an `RDD` of Strings, each String contains one *token* (as defined by our simple tokenizer).

With that function, we can create an RDD of tokens and finally count words. There are probably lots of ways of doing it, here is one of those:

{% highlight scala %}
val tokens = tokenize(book)

val tokenAndCount =
  tokens
    .map(token => (token, 1))
    .reduceByKey(_ + _)
{% endhighlight %}

On the code above we are creating a value with our tokens and then for each token, we are mapping it to a tuple consisting of the token and the number 1. After that, we are using the `reduceByKey` method that comes implicitly from the `PairRDDFunctions`. This method merges the values of the same key using reducing function, in this case, a simple sum (`_ + _`).

Ultimately, we need to save this work. We can easily do this with the method `saveAsTextFile`:

{% highlight scala %}
tokenAndCount.saveAsTextFile("./data/output")
{% endhighlight %}

This will save the output of our job on the defined directory. Note that Spark will not save if the directory is already there, so make sure to delete or move the output directory between runs, or maybe add a timestamp to the name of the directory to make sure we will have no collisions. The final state of our code at this point is:

{% highlight scala %}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

object Boot {

  val numberOfCores = 4

  def tokenize(text: RDD[String]): RDD[String] =
    text.flatMap(_.split(" "))

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Spark Seed")
      .setMaster(s"local[$numberOfCores]")

    implicit val sc = new SparkContext(conf)

    val book = sc.textFile("./data/input/pg1342.txt")

    val tokens = tokenize(book)

    val tokenAndCount =
      tokens
        .map(token => (token, 1))
        .reduceByKey(_ + _)

    tokenAndCount.saveAsTextFile("./data/output")

    sc.stop()
  }
}
{% endhighlight %}

# Adding Tests

So we now need a simple way of testing our tokenize function without running the job, so, let's create a test on the test directory with the name `TokenizerSpec.scala`:

{% highlight scala %}
import com.holdenkarau.spark.testing.SharedSparkContext
import org.scalatest.{Matchers, FlatSpec}

class TokenizerSpec extends FlatSpec with Matchers with SharedSparkContext {

  behavior of "TokenizerSpec"

  "tokenize" should "tokenize  ¯\\_(ツ)_/¯." in {

    val text = sc.parallelize(Seq("This   is   a line", "This is another line"))

    val tokens = Boot.tokenize(text).collect().toSeq

    tokens should be (Seq("This", "is", "a", "line", "This", "is", "another", "line"))

  }

}
{% endhighlight %}

Here we are using the helper `SharedSparkContext` that we added as a dependency in the beginning of this post, this trait adds a `SparkContext` to your test class under the name `sc`. After calling our `tokenize` function we are collecting the results and transforming it to a `Seq`, for tests it is ok, however, be careful when using collect since it will load all the data into the driver, which may end in OutOfMemory exceptions.

If we go now to our sbt console and type the command `test` we will see the following:

{% highlight bash %}
> test
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/Users/ftruzzi/.ivy2/cache/ch.qos.logback/logback-classic/jars/logback-classic-1.1.3.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/Users/ftruzzi/.ivy2/cache/org.slf4j/slf4j-log4j12/jars/slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [ch.qos.logback.classic.util.ContextSelectorStaticBinder]
[info] TokenizerSpec:
[info] TokenizerSpec
[info] tokenize
[info] - should tokenize  ¯\_(ツ)_/¯. *** FAILED ***
[info]   Array("This", "", "", "is", "", "", "a", "line", "This", "is", "another", "line") was not equal to List("This", "is", "a", "line", "This", "is", "another", "line") (TokenizerSpec.scala:14)
[info] ScalaCheck
[info] Passed: Total 0, Failed 0, Errors 0, Passed 0
[info] ScalaTest
[info] Run completed in 2 seconds, 142 milliseconds.
[info] Total number of tests run: 1
[info] Suites: completed 1, aborted 0
[info] Tests: succeeded 0, failed 1, canceled 0, ignored 0, pending 0
[info] *** 1 TEST FAILED ***
[error] Failed: Total 1, Failed 1, Errors 0, Passed 0
[error] Failed tests:
[error] 	TokenizerSpec
[error] (test:test) sbt.TestsFailedException: Tests unsuccessful
[error] Total time: 3 s, completed Mar 16, 2016 12:12:37 AM
>
{% endhighlight %}

We found a problem! Our tokenizer function is actually adding empty strings to our result. So we go back to our code and fix our functions:

{% highlight scala %}
def tokenize(text: RDD[String]): RDD[String] =
  text
    .flatMap(_.split(" "))
    .filter(_.nonEmpty)
{% endhighlight %}

# Running it

If you didn't run it already, we have some options, for debugging we can just hit `run`, or we can use the plugin we defined in the beginning on our `build.sbt`. So let's just call the `stage` command on the sbt console:

{% highlight bash %}
> stage
[info] Packaging /Users/ftruzzi/devel/spark-seed/target/scala-2.11/spark-seed_2.11-0.1-sources.jar ...
[info] Wrote /Users/ftruzzi/devel/spark-seed/target/scala-2.11/spark-seed_2.11-0.1.pom
[info] Done packaging.
[info] Compiling 1 Scala source to /Users/ftruzzi/devel/spark-seed/target/scala-2.11/classes...
[info] Main Scala API documentation to /Users/ftruzzi/devel/spark-seed/target/scala-2.11/api...
model contains 2 documentable templates
[info] Packaging /Users/ftruzzi/devel/spark-seed/target/scala-2.11/spark-seed_2.11-0.1.jar ...
[info] Done packaging.
[info] Main Scala API documentation successful.
[info] Packaging /Users/ftruzzi/devel/spark-seed/target/scala-2.11/spark-seed_2.11-0.1-javadoc.jar ...
[info] Done packaging.
[success] Total time: 3 s, completed Mar 16, 2016 12:19:05 AM
{% endhighlight %}

This created some files under the directory `./target/universal/stage/`. One of them is the `bin` directory that contains two executable shell scripts one for unix like and other for windows systems. The other directory `lib` contains all our dependencies, so if we want to deploy this somewhere you could copy this stage directory and it will work in any machine that contains a JVM.

Finally we can run our code with:
{% highlight bash %}
➜  spark-seed git:(master) ✗ ./target/universal/stage/bin/spark-seed
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/Users/ftruzzi/devel/spark-seed/target/universal/stage/lib/ch.qos.logback.logback-classic-1.1.3.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/Users/ftruzzi/devel/spark-seed/target/universal/stage/lib/org.slf4j.slf4j-log4j12-1.7.10.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [ch.qos.logback.classic.util.ContextSelectorStaticBinder]
{% endhighlight %}

After that we will end up with a new directory under the data directory `output`, with the following structure:

{% highlight bash %}
➜  spark-seed git:(master) ✗ tree data/output
data/output
├── _SUCCESS
├── part-00000
└── part-00001
{% endhighlight %}

The `_SUCCESS` is a hadoop convention that indicates that the job was successful and it saved its output there. The other files contains your actual result:

{% highlight bash %}
➜  spark-seed git:(master) ✗ head data/output/part-00000
(dare.",2)
("Pardon,1)
(intimately,2)
(House,1)
(muslin,,1)
(Never,,2)
(park,,4)
(insufferably,1)
(winter;,2)
(Epsom,,1)
{% endhighlight %}

With this output we see that we should probably add more tests to our tokenizer function, that's on you!

The complete code used here is available on the [`spark-seed` repo](https://github.com/flaviotruzzi/spark-seed).

Cheers!
