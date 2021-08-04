#encoding:utf-8
import tensorflow as tf
class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=100000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=600         #max length of sentence
    num_classes=8       #number of labels

    num_filters=128        #number of convolution kernel
    filter_sizes=[2, 3, 4]   #size of convolution kernel

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

class TextCNN(object):

    def __init__(self,config):

        self.config=config

        self.input_x=tf.compat.v1.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
        self.input_y=tf.compat.v1.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
        self.keep_prob=tf.compat.v1.placeholder(tf.float32,name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.l2_loss = tf.constant(0.0)

        self.cnn()
    def cnn(self):

        with tf.device('/cpu:0'):
            self.embedding = tf.compat.v1.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.compat.v1.constant_initializer(self.config.pre_trianing))
            self.embedding_inputs= tf.nn.embedding_lookup(params=self.embedding, ids=self.input_x)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1)

        with tf.compat.v1.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.compat.v1.name_scope("conv-maxpool-%s" % filter_size):

                    filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                    W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        input=self.embedding_inputs_expanded,
                        filters=W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    pooled = tf.nn.max_pool2d(
                        input=h,
                        ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.outputs= tf.reshape(self.h_pool, [-1, num_filters_total])


        with tf.compat.v1.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.outputs, 1 - (self.keep_prob))

        with tf.compat.v1.name_scope('output'):
            fc_w = tf.compat.v1.get_variable('fc_w', shape=[self.final_output.shape[1], self.config.num_classes],
                                   initializer=tf.compat.v1.keras.initializers.glorot_normal())
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            self.logits = tf.matmul(self.final_output, fc_w) + fc_b
            self.prob=tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(input=self.logits, axis=1, name='predictions')

        with tf.compat.v1.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=tf.stop_gradient(self.input_y))
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            self.loss = tf.reduce_mean(input_tensor=cross_entropy) + self.config.l2_reg_lambda * self.l2_loss
            self.loss = tf.reduce_mean(input_tensor=cross_entropy)

        with tf.compat.v1.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

        with tf.compat.v1.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(input=self.input_y,axis=1),self.y_pred_cls)
            self.acc=tf.reduce_mean(input_tensor=tf.cast(correct_pred,tf.float32))


