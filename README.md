Classification of NLTK datasets using scikit-learn

In this project I load 3 different corpora available in the nltk package, and process and classify them with models available in 
scikit-learn.

The data run on are the Inaugural corpus (every US presidential inauguration speech up to Obama 2009), the Reuters corpus (1.3-million 
word corpus contaning news articles), and the Gutenberg corpus (large collection of text from famous books freely available online). For 
each of these corpora, the training examples are individual sentences, and the classes are the exact documents/files that the sentence is
found in (for Inaugural, this corresponds to the presidential inauguration (president+year), for Reuters, the particular news article, 
and for Gutenberg, the specific book). Data is preprocessed by converting to one-hot vectors, as is standard in preparation of text data 
for machine learning. Once processed, it is then run through 3 different classification algorithms: Logistic Regression, Naive Bayes, and 
Linear SVC. (I made a K-Neighbors as well, but I've left that commented out due to its not handling larger datasets as well as the other 
models, as well as the fact that KNN isn't the best choice of algorithm for text classification).

The main script, skl_utt_classifier.py, needs 2 CSV files passed as arguments, the first being the training file and the 
second the test file. The other script in this repo, extract_from_corpus.py, requires a string passed as an argument, which may be one of 
"inaugural", "reuters", or "gutenberg". This script then runs over the given NLTK corpus and organizes the data into sentence-label pairs, 
and outputs both a training and a test file in CSV format (the test file holds out 20% of the total sentences in the corpus). The files 
corpus_labeled_<name>_train.csv and corpus_labeled_<name>_test.csv for Inaugural, Reuters, and Gutenberg are already provided in this 
repo, and are outputs of this script. To load any other datasets from nltk, it is necessary to run extract_from_corpus.csv passing the 
corpus name as an argument.

Enhancements to the performance of the models on these and more corpora are pending. I did attempt a stopword removal step in 
preprocessing, but this actually worsened performance slightly, so I took out this step, but have left it commented in the code for 
reference and possible future improvement.

This is my first attempt at building my own machine learning models after having taken Andrew Ng's machine learning and deep learning 
courses on Coursera. I elected to combine the most common and effective classification models available in scikit-learn with the very well-organized and robust corpora available in NLTK. The results are high-performing baseline models (for Linear SVC, 99% for Inaugural 
and 92% for Gutenberg, the largest corpus) that will only improve with fine-tuning of features, steps taken in data preprocessing, and 
parameters within the scikit-learn models themselves.