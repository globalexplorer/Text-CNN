#encoding:utf-8
import logging
import time
import codecs
import sys
import re
import pkuseg
from gensim.models import word2vec
from text_model import TextConfig



class Get_Sentences(object):
    '''

    Args:
         filenames: a list of train_filename,test_filename,val_filename
    Yield:
        word:a list of word cut by pkuseg

    '''


    def __init__(self, filenames):
        stopWords_fn = './data/stopwords.txt'
        self.stopWords_set = self.get_stopWords(stopWords_fn)
        self.seg = pkuseg.pkuseg(model_name = "default", user_dict = "general_dict.txt") 

        self.filenames= filenames
    
    def get_stopWords(self, stopWords_fn):
        with open(stopWords_fn, 'r', encoding='utf-8') as f:
            stopWords_set = {line.strip() for line in f}
        return stopWords_set
    
    def sentence2words(self, sentence, stopWords=False, stopWords_set=None):
        """ 
        split a sentence into words based on pkuseg
        """

        seg_words = self.seg.cut(sentence)
        if stopWords:
            words = [word for word in seg_words if word not in stopWords_set and word != ' ']
        else:
            words = [word for word in seg_words]
        return words
    
    def __iter__(self):
        for filename in self.filenames:
            with codecs.open(filename, 'r', encoding='utf-8') as f:
                for _,line in enumerate(f):
                    try:
                        line=line.strip()
                        line=line.split('^')
                        assert len(line)==2 
                        if len(line[1])>50:
                            yield self.sentence2words(line[1], True, self.stopWords_set)
                    except:
                        pass
                    
    
    

def train_word2vec(filenames):
    '''
    use word2vec train word vector
    argv:
        filenames: a list of train_filename,test_filename,val_filename
    return: 
        save word vector to config.vector_word_filename

    '''
    t1 = time.time()
    sentences = Get_Sentences(filenames)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(sentences, size=config.embedding_size, window=5, min_count=10, max_vocab_size=config.vocab_size, workers=10)
    model.save('word2vec.model')

    model.wv.save_word2vec_format(config.vector_word_filename, binary=False)
    print('-------------------------------------------')
    print("Training word2vec model cost %.3f seconds...\n" % (time.time() - t1))


if __name__ == '__main__':
    config=TextConfig()
    filenames=[config.train_filename,config.test_filename,config.val_filename]
    train_word2vec(filenames)

