import pandas as pd
import tensorflow as tf

TRAIN_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data'
TEST_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data'

CSV_COLUMN_NAMES = [
    'card_1_suit',
    'card_1_rank',
    'card_2_suit',
    'card_2_rank',
    'card_3_suit',
    'card_3_rank',
    'card_4_suit',
    'card_4_rank',
    'card_5_suit',
    'card_5_rank',
    'poker_hand'
]

POKER_HANDS = [
    'Nothing',
    'One pair',
    'Two pairs',
    'Three of a kind',
    'Straight',
    'Flush',
    'Full house',
    'Four of a kind',
    'Straight flush',
    'Royal flush'
]

def maybe_download():
    """Use keras to download the datasets if they aren't already downloaded"""
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='poker_hand'):
    """Returns the poker hand dataset as (train_x, train_y), (test_x, test_y)"""
    train_path, test_path = maybe_download()

    # Load training data as a pandas DataFrame.
    # x refers to features and y refers to labels.
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    # Load test data as a pandas DataFrame.
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    """Input function for training the network"""
    # Convert the inputs into a TensorFlow DataSet.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # We are predicting, so only use features.
        inputs = features
    else:
        # We are evaluating, so use both.
        inputs = (features, labels)


    # Convert the inputs into a TensorFlow Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    dataset = dataset.batch(batch_size)

    return dataset