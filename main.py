import os
import pprint
import tensorflow as tf

#from data import read_data
from data import *
from absa_model import MemN2N_ABSA

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
flags.DEFINE_integer("nhop", 6, "number of hops [6]")
flags.DEFINE_integer("mem_size", 100, "memory size [100]")
flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
flags.DEFINE_integer("nwords", -1, "number of words in training set [-1]")
# verify the labels numbers in sentiment problem -- only 3 (originally 4)
flags.DEFINE_integer("nlabels", 3, "number of labels in sentiment [-1]")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
flags.DEFINE_string("data_dir", "data", "data directory [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string("data_name", "ptb", "data set name [ptb]")
flags.DEFINE_boolean("is_test", False, "True for testing, False for Training [False]")
flags.DEFINE_boolean("show", False, "print progress [False]")
flags.DEFINE_string("glove_embedn", "../datasets_senti/glove.42B.300d.zip", "glove embeddings path")
flags.DEFINE_string("xml_data_dir", "../datasets_senti/Aspect-Based-Sentiment-Analysis/datasets/ABSA-SemEval2014/", "xml data dir")
flags.DEFINE_string("laptop_train", "Laptop_Train_v2.xml", "Laptop train xml")
flags.DEFINE_string("laptop_test", "ABSA_Gold_TestData/Laptops_Test_Gold.xml", "Laptop test xml")

FLAGS = flags.FLAGS

def main(_):
    count = []
    word2idx = {}

    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)

    # the code below reads the review datasets and glove embedding to
    # to get 1) data in terms of word_indxes, 2) glove embeddings in
    # terms of num_trainNtest_words x 300 array
    embed_dict = get_embedding_dict(FLAGS.glove_embedn)
    indxd_embedn = []; embedn_count = [];
    embedn_word2idx = {}; multiwrd_indx_cmpnnt = {};
    max_sent_len, data, trgt_aspect, trgt_Y, trgt_pos = get_dataNembedding( \
        '%s/%s' % (FLAGS.xml_data_dir, FLAGS.laptop_train), embed_dict, \
        indxd_embedn, embedn_count, embedn_word2idx, multiwrd_indx_cmpnnt)
    # 20% random split to train and valid
    train_data, train_trgt_aspect, train_trgt_Y, train_trgt_pos, \
        valid_data, valid_trgt_aspect, valid_trgt_Y, valid_trgt_pos \
        = split_data(data, trgt_aspect, trgt_Y, trgt_pos)
    max_sent_len_test, test_data, test_trgt_aspect, test_trgt_Y, \
        test_trgt_pos = get_dataNembedding( '%s/%s' % \
        (FLAGS.xml_data_dir, FLAGS.laptop_test), embed_dict, \
        indxd_embedn, embedn_count, embedn_word2idx, multiwrd_indx_cmpnnt)

    #cnsldtd_aspect = (train_trgt_aspect, valid_trgt_aspect, test_trgt_aspect)
    embed_nparray = get_nparray_embedding(embed_dict, \
        FLAGS.init_std, embedn_word2idx, multiwrd_indx_cmpnnt)
    FLAGS.nwords = len(embedn_word2idx)
    FLAGS.edim = embed_nparray.shape[1]
    #TODO: check the above
    #NOTE: the max sent len aboev doesnt incude the aspect word
    FLAGS.mem_size = max(max_sent_len, max_sent_len_test)
    train_mask = get_context_maskNpad(train_data, FLAGS.mem_size)
    test_mask = get_context_maskNpad(test_data, FLAGS.mem_size)
    valid_mask = get_context_maskNpad(valid_data, FLAGS.mem_size)
    #labels_dict = zip(["negative","neutral","positive"], xrange(0,3))
    labels_dict = dict(zip(["conflict","negative","neutral","positive"], \
            xrange(0,4)))
    FLAGS.nlabels = 4
    #TODO: convert the data to only onclude 3 classes




    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N_ABSA(FLAGS, sess, embed_nparray, labels_dict)
        model.build_model()
        #writer = tf.train.SummaryWriter("./basic", tf.get_default_graph().as_graph_def())
        writer = tf.summary.FileWriter("./basic", tf.get_default_graph().as_graph_def())
        writer.flush()
        cnsldted_valid_data = (valid_data, valid_trgt_aspect, valid_trgt_Y, \
                valid_mask)
        cnsldted_train_data = (train_data, train_trgt_aspect, train_trgt_Y, \
                train_mask)
        cnsldted_test_data = (test_data, test_trgt_aspect, test_trgt_Y, \
                test_mask)

        if FLAGS.is_test:
            model.run(cnsldted_valid_data, cnsldted_test_data)
        else:
            model.run(cnsldted_train_data, cnsldted_valid_data)

if __name__ == '__main__':
    tf.app.run()
