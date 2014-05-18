CS909 Coursework
================

This coursework uses python 2.7 and requires the following libraries:

* BeautifulSoup4 - used for XML parsing
* NLTK (+ downloaded corpora) - used for pre-processing
* GenSim - used for topic modelling
* scikit-learn - classification / clustering

Execution
=========

Having downloaded all the required libraries, the following functions can be run.

``python load.py``: This will perform the preprocessing of the Reuters dataset. This has already been performed and a processed copy of the dataset is included.

``python learn.py``: This will run all the parts of the project including k-cross fold validation, testing the best model and applying three clustering algorithms. By default, TF*IDF is used and k=10 in cross fold validation.

Options
-------

The learn.py script can also be run with the following options which enables other implemented features to run.

``python learn.py --LDA``: Uses LDA topic modelling for feature reduction.

``python learn.py --count``: Uses the term frequency rather than TF*IDF.

``python learn.py --binary``: Uses binary frequency for the existance of a term in a document.

``python learn.py --k 10``: Change the number of folds to use in k-cross fold validation.
