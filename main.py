# encoding:utf-8
import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow import feature_column as fc
import feature_processing as fe
from feature_processing import FeatureConfig
import model
import config

FLAGS = config.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuid

if FLAGS.run_on_cluster:
    cluster = json.loads(os.environ["TF_CONFIG"])
    task_index = int(os.environ["TF_INDEX"])
    task_type = os.environ["TF_ROLE"]


def main(unused_argv):
    feature_configs = FeatureConfig().create_features_columns()
    classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                                                      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                                                      keep_checkpoint_max=3),
                                        params={"feature_configs": feature_configs,
                                                "learning_rate": FLAGS.learning_rate}
                                        )
    def train_eval_model():
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: fe.train_input_fn(FLAGS.train_data, FLAGS.batch_size),
                                            max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: fe.eval_input_fn(FLAGS.eval_data, FLAGS.batch_size),
                                          start_delay_secs=60,
                                          throttle_secs = 30,
                                          steps=1000)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    def train_model():
        from tensorflow.python import debug as tf_debug
        debug_hook = tf_debug.LocalCLIDebugHook()
        classifier.train(input_fn=lambda: fe.train_input_fn(FLAGS.train_data, FLAGS.batch_size), steps=1000, hooks=[debug_hook,])

    def eval_model():
        classifier.evaluate(input_fn=lambda: fe.eval_input_fn(FLAGS.eval_data, FLAGS.batch_size), steps=1000)

    def export_model(inputs_dict, model_path):
        feature_spec = feature_configs.feature_spec
        feature_map = {}
        for key, feature in feature_spec.items():
            if key not in inputs_dict:
                continue
            if isinstance(feature, tf.io.VarLenFeature):
                feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[1], name=key)
            elif isinstance(feature, tf.io.FixedLenFeature):
                feature_map[key] = tf.placeholder(dtype=feature.dtype, shape=[None, feature.shape[0]], name=key)
        serving_input_recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        export_dir = classifier.export_saved_model(model_path, serving_input_recevier_fn)
        print("pb model exported to %s"%export_dir)

    if FLAGS.train_eval_model:
        train_eval_model()

    if FLAGS.is_eval:
        eval_model()

    if FLAGS.export_user_model:
        inputs_dict = {"userID": True}
        export_model(inputs_dict, FLAGS.user_model_path)

    if FLAGS.export_item_model:
        FLAGS.export_user_model = False
        inputs_dict = {"itemID": True}
        export_model(inputs_dict, FLAGS.item_model_path)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
