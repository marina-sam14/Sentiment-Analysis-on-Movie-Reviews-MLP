
# MLPs-Classification
The first model architecture consists of three densely connected layers, with rectified linear unit (ReLU) activation functions for the hidden layers and a sigmoid activation function for the output layer. The Adam optimizer is employed with a learning rate of 0.001, and the binary cross-entropy loss function is used for training. The training process is configured to run for 100 epochs with a batch size of 256, and the model weights are saved at each epoch if the validation accuracy improves. The training time is recorded, and the resulting model is designed for binary classification tasks.

## Tuning
The utilization of the [Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) library is central to our hyperparameter tuning process. In our initial attempt to fine-tune the model, we experimented with a configuration involving three dense layers. However, due to observed oscillations in the results, we retained that code within comments and opted for a streamlined approach featuring two dense layers. These layers incorporate rectified linear unit (ReLU) activation functions, dropout layers for regularization, and a sigmoid activation function in the output layer. It's important to note that we made an attempt with the softmax activation function in the output layer. However, during experimentation, we observed that the accuracy consistently hovered around 50\%. This outcome led us to reconsider the choice of activation function, ultimately opting for the sigmoid activation. The sigmoid function proved more suitable for the binary classification problem at hand, demonstrating improved performance and better capturing the nuances of the task. The hyperparameters subject to tuning include the number of units in each dense layer, dropout rates, and the learning rate for the Adam optimizer. Utilizing a RandomSearch methodology, we explored 5 distinct combinations of these hyperparameters. Subsequent to the hyperparameter search, we constructed, trained, and evaluated the performance of the best model on both training and validation datasets. To visually depict the training progress, we generated plots illustrating accuracy and loss curves. It is noteworthy that we concluded the training process prematurely at 70 epochs, identifying potential overfitting trends in the loss curves.

## Performance
We assess the model's performance using standard evaluation metrics such as precision, recall, and F1 score, providing a comprehensive understanding of its ability to correctly classify positive and negative sentiments.

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






