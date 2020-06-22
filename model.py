# coding:utf-8
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow.python.ops import partitioned_variables
import config
import feature_processing as fe

FLAGS = config.FLAGS

def build_user_model(features, mode, params):
    with tf.variable_scope("user_side", partitioner=tf.fixed_size_partitioner(len(FLAGS.ps_hosts.split(",")), axis=0)):
        user_did_embed = fc.input_layer(features, params["feature_configs"].all_columns["userID"])
        user_dense = tf.nn.l2_normalize(user_did_embed)
        return user_dense

def build_item_model(features, mode, params):
    with tf.variable_scope("item_side", partitioner=tf.fixed_size_partitioner(len(FLAGS.ps_hosts.split(",")), axis=0)):
        item_uuid_embed = fc.input_layer(features, params["feature_configs"].all_columns["itemID"])
        item_dense = tf.nn.l2_normalize(item_uuid_embed)
        return item_dense

def model_fn(features, labels, mode, params):
    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        # 导出user和item向量
        if FLAGS.export_user_model:
            user_net = build_user_model(features, mode, params)
            predictions = {"user_vector": user_net}
        elif FLAGS.export_item_model:
            item_net = build_item_model(features, mode, params)
            predictions = {"item_vector": item_net}
        export_outputs = {"prediction": tf.estimator.export.PredictOutput(outputs=predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    user_net = build_user_model(features, mode, params)
    item_net = build_item_model(features, mode, params)
    # MF
    gmf_layer = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keepdims=True, name="gmf")
    
    # MLP
    mlp_layer = tf.concat([user_net, item_net], axis=1)
    mlp_layer = tf.layers.dense(mlp_layer, units=128, activation=tf.nn.leaky_relu, name="mlp_layer_1")
    mlp_layer = tf.layers.dense(mlp_layer, units=32, activation=tf.nn.leaky_relu, name="mlp_layer_2")
    mlp_layer = tf.layers.dense(mlp_layer, units=1, activation=None, name="mlp_layer_3")
    
    # NCF
    net = tf.concat([gmf_layer, mlp_layer], axis=1)
    predict_layer = tf.layers.dense(net, units=1, name="model_output")
    pred = tf.sigmoid(predict_layer, name="scores")

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.cast(labels, tf.float32)
        loss = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)) +
                               (1.0 - labels) * tf.log(tf.clip_by_value(1.0 - pred, 1e-10, 1.0)))
        metrics = {"auc": tf.metrics.auc(labels=labels, predictions=pred)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.cast(labels, tf.float32)
        loss = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)) +
                               (1.0 - labels) * tf.log(tf.clip_by_value(1.0 - pred, 1e-10, 1.0)))
        global_step = tf.train.get_global_step()
        train_op = (tf.train.AdagradOptimizer(0.005).minimize(loss, global_step=global_step))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
