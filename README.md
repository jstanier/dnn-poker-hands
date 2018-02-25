# dnn-poker-hands
Using deep learning to predict the quality of poker hands. The data set is the [Poker Hand Data Set](http://archive.ics.uci.edu/ml/datasets/Poker+Hand)

This was created mostly so I could give myself a go at learning the high-level [TensorFlow APIs](https://www.tensorflow.org/), namely Estimators and Datsets. 

## Installing dependencies

This assumes you have virtualenv with support for Python 3 installed.

```
./setup.sh
```

## Training and evaluating the model

You can run it using default parameters like so:

```
python estimator.py ********

```

## Results

I don't know much about how to scientifically test and tune neutral networks (yet). But here are some experimental results against the test set.

* 1 hidden layer of 10 neurons, 1000 training steps: 0.500
* 1 hidden layer of 1000 neurons, 50000 training steps: 0.705
* 2 hidden layers of 10 neurons, 1000 training steps: 0.504
* 2 hidden layers of 1000 neurons, 50000 training steps: 0.882
* 5 hidden layers of 10 neurons, 50000 training steps: 0.640
* 5 hidden layers of 100 neurons, 50000 training steps: 0.954
* 10 hidden layers of 100 neurons, 50000 training steps: 0.694

The [original paper](https://eembdersler.files.wordpress.com/2010/09/2014913024-gokaydisken-project.pdf) achieved 0.924.