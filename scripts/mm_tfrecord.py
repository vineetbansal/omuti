import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    with tf.io.TFRecordWriter('scratch/example.tfrecords') as writer:
      for _ in range(4):
        x, y = np.random.random(), np.random.random()

        o = tf.train.Example(features=tf.train.Features(feature={
            "x": tf.train.Feature(float_list=tf.train.FloatList(value=[x])),
            "y": tf.train.Feature(float_list=tf.train.FloatList(value=[y])),
        }))

        writer.write(o.SerializeToString())

    dataset = tf.data.TFRecordDataset(['scratch/example.tfrecords'])
    for raw_record in dataset.take(10):
      o = tf.train.Example()
      o.ParseFromString(raw_record.numpy())
      print(o.features.feature['x'].float_list.value[0] + 1)  # we get a float