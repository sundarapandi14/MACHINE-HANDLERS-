import numpy as np
import pandas as pd
import tensorflow as tf
dataset =  pd.read_csv ("crops.csv")
CSV_COLUMN_NAMES = ['Mintemp', 'Maxtemp,
                    'Minrainfall', 'Maxrainfall', 'Sesons']

"""def input_evaluation_set():
    features = {'mintemp': np.array([15.0,12.0]),
                'maxtemp':  np.array([27.5,25.5]),
                'minrainfall': np.array([100.0,25.0]),
                'maxrainfall':  np.array([150.0,75.0,])}
    labels = np.array([2, 1])
    return features, labels"""
def train_input_fn(features, labels, batch_size):
    
  """An input function for training"""
dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    # Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
classifier.train(
    input_fn=lambda:iris_data.
    train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)
eval_result=classifier.evaluate(
        input_fn=lambda:.eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
predict_x = {
if (season="summer",mintemp<="15.0" and maxtemp>="27.0",MAXRAINFALL<="150" and MINRAINFALL>="100")
    print("Rice","wheat","maize","millets","bajra");
    elif(season="winter",mintemp<="20.0" and maxtemp>="27.0",MAXRAINFALL<="60"and MINRAINFALL>="25")
    print("pulses","lentil","oilseeds","groundnut");
    elif(season="autumn",mintemp<="20.0" and maxtemp>="35.0",MAXRAINFALL<="165" and MINRAINFALL>="85")
    print("sugarcane","sugarbeet","cotton","tea","coffee");
    elif(season="monsoon",mintemp<="18.0" and maxtemp>="35.0" ,MAXRAINFALL<="250" and MINRAINFALL>="100")
    print("cocoa","rubber","jute","coconut");
    elif(season="spring",mintemp<="10.0"and maxtemp>="40.0",MAAXRAINFALL<="400"and MINRAINFALL>="15")
    print("flax","oilpalm","clove","black pepper","cardamom","turmeric");
    }
)

  