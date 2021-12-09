# Prediction of Usefulness of Written Reviews on Steam ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## Goal
The goal of this project is to predict how useful a written review is on [Steam](https://store.steampowered.com/). The quantitative target is detailed below.

## Data Source

The data is downloaded from https://github.com/mulhod/steam_reviews/tree/master/data with [GitZip](https://github.com/KinoLien/gitzip).

### Prediction Target
The target is what I call **usefulness**.
Since the data include not only the written reviews but also metadata such as the reviewers' information, there are also the number of votes that people think the review is useful, and the total number of votes. I thus define  
```usefulness = (number of votes for being useful)/(number of total votes)```

## Machine Learning Algorithms Used for Prediction
There are different ways of utilizing the data to predict the target value.  
For a more primitive approach of using only the metadata, see an earlier study [here](https://github.com/kaikai581/steam-reviews/blob/master/Usefulness%20Regression.ipynb).  

Below I will talk about only using the written review to predict the target value.

### Data Preprocessing
In order to use records that is representative of users' perception, reviews with a total number of votes no greater than 10 are dropped. Then, the remaining reviews are separated into two the training set and the test set with a 3:1 ratio.  

### GLoVe: GLobal Vectors Word Embeddings
This project uses the GLoVe word embeddings method to vectorize the written reviews.  

Unlike the more primitive ways such as Bag of Words, which represents a document as a histogram of frequencies, GLoVe uses a continuous vector space of a fixed dimension where each dimension represent some meaning of a word. The meanings are learned from unsupervised algorithms such that words with similar meanings are close by.  

In this project, I have downloaded the pretrained model from the [official website](https://nlp.stanford.edu/projects/glove/) ([glove.6B.zip](https://nlp.stanford.edu/data/glove.6B.zip)), and vectorize each review by the following steps.  
1. Tokenize each review
2. Get a vector representation for each word in the tokens
3. Get a vector mean of all word vectors in a review as the document vector

### Apply Regression Algorithms to Document Vectors
In the `revisit_scripts`, there are standalone python scripts, each representing a regression model I had played with.  
Each of the scripts adds a predicted value column with an appropriate column name to the csv file `processed_data/test_pred.csv` for further analysis.  

## Performance Summary
I have published the results of this project to [Heroku](https://www.heroku.com/). For details, and for the link to the published page, please consult the [GitHub page](https://github.com/kaikai581/tdi-capstone-fall-2021) for Heroku deployment.
