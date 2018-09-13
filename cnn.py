import numpy as np
import os
import re
import pickle
from tensorflow.contrib import learn

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(path):
	##load data from IMDB
    # positive_path = path + '/pos/'
    # negative_path = path + '/neg/'
    # positive_files = os.listdir(positive_path)
    # negative_files = os.listdir(negative_path)
    
    # # Load data from files
    # pos_examples = []
    # neg_examples = []
    # for fname in positive_files:
    #     positive_data_file = positive_path + fname
    #     positive_examples = list(open(positive_data_file).readlines())
    #     positive_examples = [s.strip() for s in positive_examples]
    #     pos_examples.append(positive_examples[0])
    # for fname in negative_files:
    #     negative_data_file = negative_path + fname
    #     negative_examples = list(open(negative_data_file).readlines())
    #     negative_examples = [s.strip() for s in negative_examples]
    #     neg_examples.append(negative_examples[0])
    # x_text = pos_examples + neg_examples
    # x_text = [clean_str(sent) for sent in x_text]
        
    # # Generate labels
    # positive_labels = [[0, 1] for _ in pos_examples]
    # negative_labels = [[1, 0] for _ in neg_examples]
    f = open(path,'rb')
    data = pickle.load(f)
    x_text = [s.strip() for s in data]
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in range(int(len(x_text)/2)) ]
    negative_labels = [[1, 0] for _ in range(int(len(x_text)/2)) ]
    y = np.concatenate([negative_labels, positive_labels], 0)
    return [x_text, y]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
    
def load_data(path):
    # Load and preprocess data
    # x_train, y_train = load_data_and_labels(path+'train')
    # x_test,y_test = load_data_and_labels(path+'test')
    x_train, y_train = load_data_and_labels('./data_train_list.pkl')
    x_test,y_test = load_data_and_labels('./data_test_list.pkl')

    max_document_length = max([len(x.split(" ")) for x in x_train])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.fit_transform(x_test)))
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    vocab_size = len(vocab_processor.vocabulary_)
    return x_train,y_train,x_test,y_test,vocab_size,max_document_length

import copy
import os
import math
import numpy as np
import scipy
import scipy.io
"""
This part of Dataset Generator is referenced from the pset4 in EC 500 K1
"""
class GeneratorRestartHandler(object):
    def __init__(self, gen_func, argv, kwargv):
        self.gen_func = gen_func
        self.argv = copy.copy(argv)
        self.kwargv = copy.copy(kwargv)
        self.local_copy = self.gen_func(*self.argv, **self.kwargv)
    
    def __iter__(self):
        return GeneratorRestartHandler(self.gen_func, self.argv, self.kwargv)
    
    def __next__(self):
        return next(self.local_copy)
    
    def next(self):
        return self.__next__()


def restartable(g_func):
    def tmp(*argv, **kwargv):
        return GeneratorRestartHandler(g_func, argv, kwargv)
    
    return tmp


@restartable
def dataset_generator(dataset_name,x, y, xx, yy, batch_size, epoch_n):
    assert dataset_name in ['train', 'test']
    assert batch_size > 0 or batch_size == -1  # -1 for entire dataset
    
    if dataset_name == 'train':
        X_all = x
        y_all = y
    else:
        X_all = xx
        y_all = yy
    data_len = len(X_all)
    batch_size = batch_size if batch_size > 0 else data_len
    num_batches_per_epoch = int((len(X_all)-1)/batch_size) + 1
    for epoch in range(epoch_n):
        for slice_i in range(num_batches_per_epoch):
            idx = slice_i * batch_size
            X_batch = X_all[idx:idx + batch_size]
            y_batch = y_all[idx:idx + batch_size]
            yield X_batch, y_batch

import tensorflow as tf

def cnn_map(x_,para):
    # Embedding layer
    W = tf.Variable(tf.random_uniform([para['vocab_size'], para['embedding_size']], -1.0, 1.0))
    embedded_chars = tf.nn.embedding_lookup(W, x_)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)     

    num_filters = 128
    filter_shape = [5, para['embedding_size'], 1, num_filters] #number of filters
    W_1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
    b_1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    conv1 = tf.nn.conv2d(
            embedded_chars_expanded,
            W_1,
            strides = [1,1,1,1],
            padding="VALID",
            name='conv1')
    h = tf.nn.relu(tf.nn.bias_add(conv1, b_1), name="relu")
    pool1 = tf.nn.max_pool(h,
                            ksize=[1, para['review_length'] - 5 + 1, 1, 1],
                            strides=[1,1,1,1],
                            padding='VALID',
                            name="pool")
        
    pool_flat = tf.contrib.layers.flatten(pool1, scope='pool1flat')
    dense = tf.layers.dense(inputs=pool_flat, units=500, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=2)
    return logits


def apply_classification_loss(model_function, para):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            x_ = tf.placeholder(tf.int32, [None, para['review_length']])
            y_ = tf.placeholder(tf.float32, [None,2])
            y_logits = model_function(x_,para)
            
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits)
            cross_entropy_loss = tf.reduce_mean(losses)
            trainer = tf.train.AdamOptimizer(1e-3)
            train_op = trainer.minimize(cross_entropy_loss)
            
            y_pred = tf.argmax(tf.nn.softmax(y_logits), dimension=1)
            correct_prediction = tf.equal(y_pred, tf.argmax(y_, dimension=1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model_dict = {'graph': g, 'inputs': [x_, y_], 'train_op': train_op,
                  'accuracy': accuracy, 'loss': cross_entropy_loss}
    
    return model_dict

def train_model(model_dict, dataset_generators, batch_size):
    with model_dict['graph'].as_default(), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        epoch_i = 0
        for iter_i,batch in enumerate(dataset_generators['train']):
            train_feed_dict = dict(zip(model_dict['inputs'], batch))
            sess.run(model_dict['train_op'], feed_dict=train_feed_dict)

            print_every = int((len(batch)-1)/batch_size) + 1
            if iter_i % print_every == 0:
                collect_arr = []
                for test_batch in dataset_generators['test']:
                    test_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                    to_compute = [model_dict['loss'], model_dict['accuracy']]
                    collect_arr.append(sess.run(to_compute, test_feed_dict))
                averages = np.mean(collect_arr, axis=0)
                fmt = (epoch_i,) + tuple(averages)
                epoch_i += 1
                print('epoch {:d}, loss: {:.3f}, accuracy: {:.3f}'.format(*fmt))

#Defube the path of Dataset
#Here we just hardcoded it in load_data
path = ' '
x, y, xx, yy, vocab_size,max_document_length = load_data(path)
para = {
    'review_length': max_document_length,
    'vocab_size': vocab_size,
    'embedding_size': 128,
}
dataset_generators = {
        'train': dataset_generator('train',x, y, xx, yy, 256, 20), #batch_size,epoch_n
        'test': dataset_generator('test',x, y, xx, yy, 256, 20)
}
model_dict = apply_classification_loss(cnn_map, para)
train_model(model_dict, dataset_generators, batch_size=256)