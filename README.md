# MLPs-Classification

## Problem at hand
Develop a part-of-speech (POS) tagger for one of the languages of the [Universal
Dependencies treebanks](http://universaldependencies.org/), using an MLP (implemented by
you) operating on windows of words (slides 35–36). Consider only the words, sentences, and
POS tags of the treebanks (not the dependencies or other annotations). Use Keras/TensorFlow
or PyTorch to implement the MLP. You may use any types of word features you prefer, but it
is recommended to use pre-trained word embeddings. Make sure that you use separate
training, development, and test subsets. Tune the hyper-parameters (e.g., number of hidden
layers, dropout probability) on the development subset. Monitor the performance of the MLP
on the development subset during training to decide how many epochs to use.
Include
experimental results of a baseline that tags each word with the most frequent tag it had in the training data; for words that were not encountered in the training data, the baseline should
return the most frequent tag (over all words) of the training data. Include in your report:

* Curves showing the loss on training and development data as a function of epochs.
* Precision, recall, F1, precision-recall AUC scores, for each class and classifier,
separately for the training, development, and test subsets.
* Macro-averaged precision, recall, F1, precision-recall AUC scores (averaging the
corresponding scores of the previous bullet over the classes), for each classifier,
separately for the training, development, and test subsets.
* A short description of the methods and datasets you used, including statistics about
the datasets (e.g., average sentence length, number of training/dev/test sentences and
words, vocabulary size) and a description of the preprocessing steps that you
performed.