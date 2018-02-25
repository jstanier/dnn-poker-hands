import argparse
import tensorflow as tf

import poker_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=50000, type=int, help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = poker_data.load_data()

    # Collect feature columns by iterating through the feature column names defined in poker_data.
    # They're then defined as TensorFlow features.
    feature_columns = []
    for key in train_x.keys():
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build a DNN, which takes the following inputs:
    # 1. The feature columns, defined above.
    # 2. The number of hidden layers, in this case 10 layers with 100 neurons each.
    # 3. The number of labels (classes) to predict. There are 10 possible in poker_data.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[100,100,100,100,100],
        n_classes=10
    )

    # Train the model.
    classifier.train(
        input_fn=lambda:poker_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps
    )

    # Evaluate the model
    eval_result = classifier.evaluate(
        input_fn=lambda:poker_data.eval_input_fn(test_x, test_y, args.batch_size)
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)