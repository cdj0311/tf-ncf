#encoding:utf-8
import os, json, codecs
import tensorflow as tf
from tensorflow import feature_column as fc
import config

FLAGS = config.FLAGS

class FeatureConfig(object):
    def __init__(self):
        self.all_columns = dict()
        self.feature_spec = dict()

    def create_features_columns(self):
        userID = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="userID",
                                                            hash_bucket_size=FLAGS.user_did_size,
                                                            dtype=tf.int64),
                                          dimension=FLAGS.embed_size,
                                          initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1, dtype=tf.float32))
        itemID = fc.embedding_column(fc.categorical_column_with_hash_bucket(key="itemID",
                                                            hash_bucket_size=FLAGS.item_uuid_size,
                                                            dtype=tf.int64),
                                           dimension=FLAGS.embed_size,
                                           initializer=tf.uniform_unit_scaling_initializer(factor=1e-5, seed=1,dtype=tf.float32))
        self.all_columns["userID"] = userID
        self.all_columns["itemID"] = itemID
        self.feature_spec = tf.feature_column.make_parse_example_spec(self.all_columns.values())

        return self


def parse_exp(example):
    features_def = dict()
    features_def["label"] = tf.io.FixedLenFeature([1], tf.int64)
    features_def["userID"] = tf.io.FixedLenFeature([1], tf.int64)
    features_def["itemID"] = tf.io.FixedLenFeature([1], tf.int64)
    features = tf.io.parse_single_example(example, features_def)
    label = features["label"]
    del features["label"]
    return features, label


def train_input_fn(filenames=None,
                   batch_size=128,
                   shuffle_buffer_size=1000):
    with tf.gfile.Open(filenames) as f:
        filenames = f.read().split()
    
    if FLAGS.run_on_cluster:
        files_all = []
        for f in filenames:
            files_all += tf.gfile.Glob(f)
        train_worker_num = len(FLAGS.worker_hosts.split(","))
        hash_id = FLAGS.task_index if FLAGS.job_name == "worker" else train_worker_num - 1
        files_shard = [files for i, files in enumerate(files_all) if i % train_worker_num == hash_id]
        files = tf.data.Dataset.list_files(files_shard)
        dataset = files.apply(tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), 
                                                                  cycle_length=4,
                                                                  buffer_output_elements=batch_size*4,
                                                                  sloppy=True))
    else:
        files = tf.data.Dataset.list_files(filenames)
        dataset = files.apply(tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), 
                                                              buffer_output_elements=batch_size*4, 
                                                              cycle_length=4,
                                                              sloppy=True))
    dataset = dataset.map(parse_exp, num_parallel_calls=4)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

def eval_input_fn(filenames=None,
                  batch_size=128):
    with tf.gfile.Open(filenames) as f:
        filenames = f.read().split()
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), buffer_output_elements=batch_size*4, cycle_length=4))
    dataset = dataset.map(parse_exp, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset


