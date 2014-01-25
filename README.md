NaiveBayes
==========

Vectorized approach to multinomial Naive Bayes binary classifier

I made [Naive Bayes classifier before](https://github.com/toshiakit/classification), but it was not vectorized. This is a new vectorized implementation based on [this page](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html).

__UPDATE__ mySpamFilter.m was added, which extends myNaiveBayes with file processing capability. It requires [Porter Stemmer](http://tartarus.org/martin/PorterStemmer/). Here is the [m-file version](http://tartarus.org/martin/PorterStemmer/matlab.txt). Change the file extension from .txt to .m for use.


How to use (myNaiveBayes.m)
-------------------------

First instantiate a Naive Bayes object. 

    nb = myNaiveBayes();
  
Then use the object to call `train` method with a training dataset

    nb.train(predictors, response);
  
Once the object is trained, you can call `predict` method to get classification for a new dataset. 

    p = nb.predict(new_predictors);

### Inputs and outputs 

`predictors` is a m x n matrix where m = number of emails and n = number of words in vocabulary that represents the word counts for each word in the vocabulary in each email. 

            word1 word2 word3 ...
    email1     0     1     0
    email2     1     0     0
    
`responses` is a m x 1 vector of binary classification where spam = 1 and ham = 0. 

            label (spam = 1, ham = 0)
    email1     1
    email2     0

How to use (mySpamFilter.m)
---------------------------

First instantiate an object.

    nb = mySpamFilter();

Then, build dataset from a local directory where you stored [SpamAssassin corpus](http://spamassassin.apache.org/publiccorpus/). 

    nb.buildDataset();
    
Next, train the model using the training set, and use the test set to evaluate it. 

    nb.buildModel();
    
Finally, use the model to classify a new email.

    predicted_class = nb.classifyEmail('emailSample1.txt')
    
### Inputs and outputs

When no inputs are specified for buildDataset() method, then it uses default values. 

* data source = 'ds_reduced', the directory where the downloaded corpus is located. The corpus is expected to be grouped into 'easy\_ham', 'hard\_ham' and 'spam' subfolders.  
* split = 0.7, meaning 70% of the dataset set will be used as training set. 
* repeat = false, meaning the split is randomized each time you run the method. Set it to true if you want to repeat the same split. 

Here is an example of using different settings: data source = 'ds_full', split = 0.8, and repeat = true.

    nb.buildDataset('ds_full',0.8,true);


Explanation
-----------

### Bayesian Theorem

We want to compute `p(class|word)` using Bayesian Theorem, which means "probability of class given a word". 

    p(class|word)= (p(word|class) * p(class)) / p(word)
    

We compare `p(spam|word)` and `p(ham|word)` and predict the class with higher probability. This means we can ignore the denominator because `p(word)` is the same for both spam and ham. 

    p(class|word)= (p(word|class) * p(class))

* `p(word|class)` is the conditional probability of word, given a class. 
* `p(class)` is the prior probability of a class.

Using independence assumption, we just multiply the `p(word|class)` over all the words in the email to come up with a joint probability `p(class|email)`, but this can lead to a floating point underflow problem. Solve it by using log, then multiplication becomes summation, i.e.,

    log(x*y) = log(x) + log(y) 

So the equation changes to:

    log(p(class|word)) = log(p(word|class)) + log(p(class))
    
### Probability Estimation


The prior `p(class)` can be estimated as follows.

    p(class) = number of emails by class / total number of training samples

The conditional probability for each word `p(word|class)` can be estimated  as follows.

    p(word|class) = count of word by class / total number of words by class

But we want to convert it to log, and division becomes subtraction.

    log(p(word|class)) = log(count of word by class) - log(total number of words by class)
    
There is a one problem: log(0) results in error, so we want to apply Laplacian smoothing by adding 1. 

    log(p(word|class)) = log(count of word by class + 1) - log(total number of words by class + 1)

Laplacian smoothing in effect adds a baseline probability for all the words, so that rare words don't get 0% or 100% probability of spam or ham. So the reasonable starting point is that a word can occur at least once in each class. Instead of using 1/1, we can use 1/size of the vocabulary. 

    log(p(word|class)) = log(count of word by class + 1) - log(total number of words by class + size of vocabulary)

### Prediction

Once we have the prior and conditional probabilities, we can predict the class of new emails as follows.

1. Extract the features and represent it in the same format as the predictor matrix we used for training. 
2. Get the conditional probabilities of each word in the emails
3. Compute the joint probabilities by adding the log conditional probabilities of each word + log prior
4. Predict the class with higher posterior probability

