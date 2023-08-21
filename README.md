Semantic Text Summarizer
==========================

This project is used to generate summary of given text using ML and NLP techniques.

Introduction
============

The is NLP and ML based project which generates semantic summary of a given text.


Methodology 
===========
The algorithm used to generate summary is as:
    TODO : discuss the algorithm

Project Structure
=================
The project contains three packages inside `src` as:
 * `base`
 * `text`
 * `util`

In package `base` there is main summarizer module.
Package `text` contains `text_processor` module and there are
different text processing methods.
Package `util` contains util module in which there are
different miscellaneous .
The project also contains `.gitignore` and
 `requirements.txt`
The `requirements.txt` contains all the required 
modules used in this project

Requirements:
=============
* Python 2.7 or more
* wiki-news word2Vec model. Model can be downloaded from
https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip


USAGE:
======
* create a virtual environment by running command``virtualenv ven``.
* start virtualenv by running `. ven/bin/activate`
* install requirements by following command
 ``pip install -r requirements.txt``
* Run command `. bin/env.py` 
* Run command `run.py word2vecModelPath filePath`
word2vecModelPath is full path of word2vec model and filePath is
the path of text file for which you want to get summary .

