#encoding:utf-8
from text_model_upgraded import *
import tensorflow as tf
import tensorflow.keras as kr
import os
import numpy as np
import pkuseg
import re
import heapq
import codecs

tf.compat.v1.reset_default_graph()


def predict(sentences):
    config = TextConfig()
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextCNN(config)
    save_dir = './checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')

    _,word_to_id=read_vocab(config.vocab_filename)
    input_x= process_file(sentences,word_to_id,max_length=config.seq_length)
    labels = {0:'公司新闻',
                1:'外汇新闻',
                2:'黄金快讯',
                3:'美股新闻',
                4:'期货新闻',
                5:'基金新闻',
                6:'行业新闻',
                7:'券商新闻'}

    feed_dict = {
        model.input_x: input_x,
        model.keep_prob: 1,
    }
    session = tf.compat.v1.Session()
    session.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    saver.restore(sess=session, save_path=save_path)
    y_prob=session.run(model.prob, feed_dict=feed_dict)
    y_prob=y_prob.tolist()
    cat=[]
    p=[]
    sec=[]
    for prob in y_prob:
        top2= list(map(prob.index, heapq.nlargest(2, prob)))
        cat.append(labels[top2[0]])
        sec.append(labels[top2[1]])
        p.append(max(prob))
    tf.compat.v1.reset_default_graph()
    
    return  cat, p, sec

def sentence_cut(sentences):
    """
    Args:
        sentence: a list of text need to segment
    Returns:
        seglist:  a list of sentence cut by jieba 

    """
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    seglist=[]
    seg = pkuseg.pkuseg(model_name = "default", user_dict = "general_dict.txt") 

    for sentence in sentences:
        words=[]
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                words.extend(seg.cut(blk))
        seglist.append(words)
    return  seglist


def process_file(sentences,word_to_id,max_length=600):
    """
    Args:
        sentence: a text need to predict
        word_to_id:get from def read_vocab()
        max_length:allow max length of sentence 
    Returns:
        x_pad: sequence data from  preprocessing sentence 

    """
    data_id=[]
    seglist=sentence_cut(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad=kr.preprocessing.sequence.pad_sequences(data_id,max_length)
    return x_pad


def read_vocab(vocab_dir):
    """
    Args:
        filename:path of vocab_filename
    Returns:
        words: a list of vocab
        word_to_id: a dict of word to id

    """
    words = codecs.open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]

if __name__ == '__main__':
    print('predict random samples in test data.... ')
    import random
    sentences=[]
    labels=[]
    confidence=[]
    with codecs.open('./data/test.csv','r',encoding='utf-8') as f:
        sample=random.sample(f.readlines(), 100)
        for line in sample:
            try:
                line=line.rstrip().split('^')
                assert len(line)==2
                sentences.append(line[1])
                labels.append(line[0])
            except:
                pass
    cat, p, sec=predict(sentences)
    for i,sentence in enumerate(sentences,0):
        if cat[i] != labels[i]:
            print ('----------------------the text-------------------------')
            print (sentence[:100]+'....')
            print ('sentence #%d'%i)
            print ('the orginal label:%s'%labels[i])
            print ('the predict label:%s'%cat[i])
            print ('the confidence:%f'%p[i])
            print ('the second label:%s'%sec[i])

