---
layout: post
title:  "Getting started with Spark"
date:   2016-03-14 00:58:59
author: Flavio Truzzi
categories: Scala Spark
tags:	scala spark
cover:  https://images.unsplash.com/photo-1439707769435-4bf7b5f265ba?fit=crop&fm=jpg&h=400&q=100&w=1450
---

I am using [Spark](http://spark.apache.org/) for quite sometime now. In this post I will explain how to setup a simple [Spark Self-Contained Application](http://spark.apache.org/docs/latest/quick-start.html#self-contained-applications) in a slightly different way than the one in the quick start documentation, IMHO, easier.

# Project Structure

If you want to just copy it just clone the [spark-seed repo](https://github.com/flaviotruzzi/spark-seed).

I will be using sbt, so our directory structure at the end should look like this:

{% highlight bash %}
spark-seed    
├── build.sbt
├── data
│   └── input
│       └── pg1342.txt
├── project
│   ├── build.properties
│   └── plugins.sbt
└── src
    └── main
        ├── resources
        │   └── logback.xml
        └── scala
            └── Boot.scala
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
  "org.json4s" %% "json4s-native" % "3.3.0",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "com.holdenkarau" % "spark-testing-base_2.11" % "1.6.0_0.3.1",
  "org.scalactic" %% "scalactic" % "2.2.6",
  "org.scalatest" %% "scalatest" % "2.2.6" % "test"
)

enablePlugins(JavaAppPackaging)
{% endhighlight %}
