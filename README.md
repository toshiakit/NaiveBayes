NaiveBayes
==========

Vectorized approach to multinomial Naive Bayes binary classifier

I made [Naive Bayes classifier before](https://github.com/toshiakit/classification), but it was not vectorized. This is a new vectorized implementation based on [this page](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html).

How to use
----------

First instantiate a Naive Bayes object. 

    nb = myNaiveBayes();
  
Then use the object to call train method with a training dataset

    nb.train(precictors, responses);
  
Once the object is trained, you can call predict method to get classification for a new dataset. 

    p = nb.predict(new_predictors);

Inputs and outputs
-------------------

predictors is a m x n matrix where m = number of emails and n = number of words in vocabulary that represents the word counts for each word in the vocabulary in each email. 

            word1 word2 word3 ...
    email1     0     1     0
    email2     1     0     0
    
responses is m x 1 vector of binary classification where spam = 1 and ham = 0. 

            resp
    email1     1
    email2     0

Explanation
-----------

We want to compute P(class|word) using Bayesian Theorem, which means "probability of class given a word". 

    p(class|word)= (p(word|class) * p(class)) / p(word)
    

We compare p(spam|word) and p(ham|word) and predict the class with higher probability. This means we can ignore the denominator because p(word) is the same for both spam and ham. 

    p(class|word)= (p(word|class) * p(class))

* p(word|class) is the conditional probability of word, given a class. 
* p(class) is the prior probability of a class.

Using independence assumption, we just multiply the p(word|class) over al the words in the email to come up with a joint probability p(class|email), but this can lead to a floating point underflow problem. Solve it by using log, then multiplication becomes summation.

    i.e. log(x*y) = log(x) + log(y) 

So the equation changes to:

    log(p(class|word)) = log(p(word|class)) + log(p(class))
    
Probability Estimation
----------------------

The prior p(class) can be estimated as follows.

    p(class) = number of emails by class / total number of training samples

The conditional probability for each word p(word|class) can be estimated  as follows.

    p(word|class) = count of word by class / total number of words by class

But we want to convert it to log, and division becomes subtraction.

    log(p(word|class)) = log(count of word by class) - log(total number of words by class)
    
There is a one problem: log(0) results in error, so we want to apply Raplace smoothing by adding 1. 

    log(p(word|class)) = log(count of word by class + 1) - log(total number of words by class + 1)
    
Prediction
-----------

Once we have the prior and conditional probabilities, we can predict the class of new emails as follows.

1. Extract the features and represent it in the same format as the predictor matrix we used for training. 
2. Get the conditional probabilities of each word in the emails
3. Compute the joint probabilites by adding the log conditional probabilities of each word + log prior
4. Predict the class with higher posterior probability

