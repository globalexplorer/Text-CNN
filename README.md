# GW Text-CNN

This repo is a simple TensorFlow implementation of the Convolutional Neural Networks for News Classification. 
It is built on cjymz886's code from Text classification with CNN and Word2vec at https://github.com/cjymz886/text-cnn

The dataset used here is financial news, including stocks, funds, precious metals, etc.

##  Updates
Added tokenization options using THULAC and PKUSEG.

## Requirements

To run the program, you need to install the following packages first:\
tensorflow\
sklearn\
gensim\
jieba\
numpy\
heapq\
codecs

## Usage

python newtrain_word2vec: Tokenize using jieba, train word embeddings using Word2Vec (vector_word.txt)

python text_train.py : Train model using the training data and the validation data

python text_test.py : Test model using the test data

python text_predict.py : Sample test data and show the incorrect predictions

python gw_predict.py : Show predictions on the unlabeled 媒体公众号 data

## Parameters

text_model.py
```
    embedding_size=100     #dimension of word embedding
    vocab_size=100000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=600         #max length of sentence
    num_classes=8       #number of labels

    num_filters=128        #number of convolution kernel
    filter_sizes=[5, 6, 7]   #size of convolution kernel

    keep_prob=0.7 
    rate=0.3               #droppout
    lr= 1e-3               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 6.0              #gradient clipping threshold
    l2_reg_lambda=0.01     #l2 regularization lambda

    num_epochs=20          #epochs
    batch_size=128         #batch_size
    print_per_batch =100   #print result

    train_filename='./data/train.csv'  #train data
    test_filename='./data/test.csv'    #test data
    val_filename='./data/validate.csv'      #validation data
    vocab_filename='./data/vocab.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file
```

## Test Result
Test Loss:   0.29, Test Acc:  92.34%
Precision, Recall and F1-Score...
```
              precision    recall  f1-score   support

        公司新闻       0.92      0.88      0.90      2406
        外汇新闻       0.96      0.94      0.95      2407
        黄金快讯       0.94      0.94      0.94      2361
        美股新闻       0.94      0.92      0.93      2434
        期货新闻       0.93      0.92      0.93      2567
        基金新闻       0.93      0.94      0.94      2367
        行业新闻       0.92      0.93      0.93      2414
        券商新闻       0.85      0.90      0.87      2333

    accuracy                           0.92     19289
   macro avg       0.92      0.92      0.92     19289
weighted avg       0.92      0.92      0.92     19289
```
Confusion Matrix...
```
[[2119    1    2   68   16   19   53  128]
 [   1 2272   82    6   29    2   13    2]
 [   1   55 2228    4   62    2    3    6]
 [  73   10    7 2248    5    7   61   23]
 [  17   21   45    5 2370    8   25   76]
 [  25    0    0    6    8 2223   11   94]
 [  11    3    2   40   17   41 2245   55]
 [  67    1    6   20   34   78   20 2107]]
 ```
Time usage:257.760 seconds...

## License
[MIT](https://choosealicense.com/licenses/mit/)