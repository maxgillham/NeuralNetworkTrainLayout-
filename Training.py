
'''
Laypout template used for NFL data train
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import tensorflow as tf
import makeTeamData
import pandas as pd
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    mainDat, mash = makeTeamData.getData()
    x, y, z, a = mainDat
    headers = makeTeamData.getHeaders()

    #Uncomment to only use passing...
    '''
    ind = 21
    del headers[ind:]
    for i in range(len(x)):
        del x[i][ind:]
    for i in range(len(z)):
        del z[i][ind:]
    '''


    #Checking length for debugging
    '''
    print('Lengths')
    print(len(x))
    print(len(y))
    print(len(z))
    print(len(a))

    print('Number of stats')
    print(len(headers))
    print(len(x[0]))
    print(len(z[0]))
    print(len(mash[1][0]))
    '''

    if len(x) != len(y) or len(z) != len(a) or len(x[0]) != len(z[0]) or len(headers) != len(z[0]):
        print('Array lengths are wrong.')
        return

    train_x = pd.DataFrame(x, columns=headers)
    train_y = pd.DataFrame(y, columns=['result'])
    test_x = pd.DataFrame(z, columns=headers)
    test_y = pd.DataFrame(a, columns=['result'])
    mashSet = pd.DataFrame(mash[1], columns=headers)

    my_feature_columns = []
    for stat in headers:
        my_feature_columns.append(tf.feature_column.numeric_column(stat))


    # Estimator
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2)

    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(mashSet, labels=None, batch_size=args.batch_size)
    )
    f = open('2017Predictions.csv', 'w+', newline='')
    writer = csv.writer(f, delimiter=',', quotechar="'")
    for i, pred in enumerate(zip(predictions)):
        #print(pred)
        #print(pred[0]['class_ids'])
        ind = pred[0]['class_ids'][0]
        pred = pred[0]
        print(mash[0][i][ind])
        print(pred['probabilities'][ind])

        out = mash[0][i][0] + ' vs. ' + mash[0][i][1] + ' -> '
        out += mash[0][i][ind] + ' with ' + str(float(pred['probabilities'][ind])*100) + '%'
        print(out)
        writer.writerow([mash[0][i][0], mash[0][i][1], mash[0][i][ind], float(pred['probabilities'][ind])*100])

    f.close()

# conversion function
def train_input_fn(features, labels, batch_size):
    # convert inputs to dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
