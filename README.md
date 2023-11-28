
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

### CV $K$-fold using `GridSearch`
We automate the process of hyperparameter tuning using Grid Search Cross-Validation and provides a summary of the results for multiple scoring metrics. The goal is to find the hyperparameter values that optimize the performance of the given estimator on the provided data.

$$CV(\hat{f}) = \frac{1}{N}\sum_n^NL(y_i, \hat{f}^{-\kappa(i)}(x_i))$$

### Taggers
* Dummy Tagger (*Unigram*)
* Logistic Regression Tagger
* Feed-forward neural network

#### Training the Neural Network
* Leaky Rectified Linear Unit, or Leaky ReLU, is a type of activation function
based on a ReLU, but it has a small slope for negative values instead of a flat
slope. The slope coefficient is determined before training, i.e. it is not learnt during
training. This type of activation function is popular in tasks where we may suffer
from sparse gradients, for example training generative adversarial networks. We
have used in our NN experimentally but it dind provide the results we needed
resulting in decreasing our accuracy.

* Batch Normalization (BatchNorm) is a technique commonly used in deep neural
networks, including Multilayer Perceptrons (MLPs), to improve training stability and
speed. However, its effectiveness can depend on the specific characteristics of
the task and the architecture being used. Part-of-Speech (POS) tagging with an
MLP is a sequence labeling task where each word in a sentence is assigned a
corresponding part-of-speech tag.

* Batch Normalization (BatchNorm) is a technique commonly used in deep neural
networks, including Multilayer Perceptrons (MLPs), to improve training stability and
speed. However, its effectiveness can depend on the specific characteristics of
the task and the architecture being used. Part-of-Speech (POS) tagging with an
MLP is a sequence labeling task where each word in a sentence is assigned a
corresponding part-of-speech tag. This encourages the model to use smaller weights 


### Data Preprocessing and Representation
So our apporach to this problem begins with our reprsentation of data. Thus we have
chosen 3 separete representation policies regarding the embeddings namely we repre-
sent each word as a word vector looking to adjacent directions. We used fasttext for our
pretrained embeddings. The window embeddings approach demonstrates high speed,
thanks to its compact dimensionality , while also exhibiting excellent performance in
terms of accuracy and f1-macro metrics. This suggests that the word vectors encap-
sulate rich and robust information. We combine 3 total methods in order to achieve an
optimal representation the classical one i.e. using classical vectorization techniques, the
embedding mehod and the boosted one which is a combination of the aformentioned i.e.
It creates feature vectors by concatenating classical and embeddings feature vectors






