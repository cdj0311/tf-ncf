# coding:utf-8
import json, os, re, codecs
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")
flags.DEFINE_boolean("is_eval", False, "Whether evaluate or not")
flags.DEFINE_boolean("export_user_model", False, "Whether export user model or not")
flags.DEFINE_boolean("export_item_model", False, "Whether export item model or not")
flags.DEFINE_boolean("train_eval_model", True, "Whether train_eval_model or not")

flags.DEFINE_string("train_dir", "", "")
flags.DEFINE_string("data_dir", "", "")
flags.DEFINE_string("log_dir", "", "")
flags.DEFINE_string("ps_hosts", "","Comma-separated list of hostname:port pairs, you can also specify pattern like ps[1-5].example.com")
flags.DEFINE_string("worker_hosts", "","Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_string("model_dir", "./ckpt/", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model/", "Saved model.")
flags.DEFINE_string("train_data", "./train_files.txt", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "./eval_files.txt", "Path to the evaluation data.")
flags.DEFINE_string("gpuid", "1", "gpuid")

flags.DEFINE_string("user_model_path", "./user_model/", "user model path")
flags.DEFINE_string("item_model_path", "./item_model/", "item model path")

flags.DEFINE_integer("train_steps",10000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 128, "Training batch size")
flags.DEFINE_integer("userID_size", 10000, "device id bucket size")
flags.DEFINE_integer("itemID_size", 10000, "uuid bucket size")
flags.DEFINE_integer("embed_size", 64, "embedding size")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("save_checkpoints_steps", 10000, "Save checkpoints every this many steps")

FLAGS = flags.FLAGS



