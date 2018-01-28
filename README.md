Data science incubator miniproject (Fall 2016): analyzing yelp data to predict venue popularity

# Overview

Our objective is to predict a new venue's popularity from information available
when the venue opens.  We will do this by machine learning from a dataset of
venue popularities provided by Yelp.  The dataset contains meta data about the
venue (where it is located, the type of food served, etc ...).  It also
contains a star rating. Note that the venues are not limited to restaurants.

## Metric

Your model will be assessed based on the root mean squared error of the number
of stars you predict.  There is a reference solution (which should not be too
hard to beat).  The reference solution has a score of 1.

## Setup cross-validation:
In order to track the performance of your machine-learning, you might want to
use
[cross_validation.train_test_split](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html).

## Submitting answers.

   To deploy a model, we suggest using the
   [`dill` library](https://pypi.python.org/pypi/dill) or
   [`joblib`](http://scikit-learn.org/stable/modules/model_persistence.html)

# Questions

##Q1: city_model
The venues belong to different cities.  You can image that the ratings in some
cities are probably higher than others and use this as an estimator.

Build an estimator that uses `groupby` and `mean` to compute the
average rating in that city.  Use this as a predictor.

**Question:** In the absence of any information about a city, what score would
you assign a restaurant in that city?

##Q2: lat_long_model
You can imagine that a city-based model might not be sufficiently fine-grained.
For example, we know that some neighborhoods are trendier than others.  We
might consider a K Nearest Neighbors or Random Forest based on the latitude
longitude as a way to understand neighborhood dynamics.

You should implement a generic `ColumnSelectTransformer` that is passed which
columns to select in the transformer and use a non-linear model like
`sklearn.neighbors.KNeighborsRegressor` or
`sklearn.ensemble.RandomForestRegressor` as the estimator (why would you choose
a non-linear model?).  Bonus points if you wrap the estimator in
`grid_search.GridSearchCV` and use cross-validation to determine the optimal
value of the parameters.

##Q3: category_model
While location is important, we could also try seeing how predictive the
venues' category. Build a custom transformer that massages the data so that it
can be fed into a `sklearn.feature_extraction.DictVetorizer` which in turn
generates a large matrix gotten by One-Hot-Encoding.  Feed this into a Linear
Regression (and cross validate it!).  Can you beat this with another type of
non-linear estimator?

