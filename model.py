from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

class Config(object):
    """RNN配置参数"""
    file_name = 'seg'  #保存模型文件

    use_embedding = False   # 是否用词向量，否则one_hot
    embedding_dim = 128      # 词向量维度
    seq_length = 50        # 序列长度
    num_classes = 5        # 类别数
    vocab_max_size = 50000       # 词汇表达小

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    # rnn = 'gru'             # lstm 或 gru

    use_crf = False
    train_keep_prob = 0.8  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 30  # 每批训练大小
    max_steps = 20000  # 总迭代batch数

    log_every_n = 20  # 每多少轮输出一次结果
    save_every_n = 100  # 每多少轮校验模型并保存


class Model(object):

    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size

        # 待输入的数据
        self.input_seqs = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input')
        self.targets = tf.placeholder(tf.int32, [None, self.config.seq_length], name='target')
        self.input_length = tf.placeholder(tf.int32, [None], name='input_length')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # 两个全局变量
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.global_loss = tf.Variable(0, dtype=tf.float32, trainable=False, name="global_loss")

        # Ann模型
        self.ann()

        # 初始化session
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def ann(self):
        """rnn模型"""
        def get_mul_cell(hidden_dim, num_layers):  # 创建多层网络
            def get_en_cell(hidden_dim):  # 创建单层网络
                enc_base_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, forget_bias=1.0)
                return enc_base_cell
            return tf.nn.rnn_cell.MultiRNNCell([get_en_cell(hidden_dim) for _ in range(num_layers)])

        def bilstm(hidm, num_layers, seq, seq_len):
            cell_fw = get_mul_cell(hidm, num_layers)
            cell_bw = get_mul_cell(hidm, num_layers)
            output, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq,sequence_length=seq_len,dtype=tf.float32)
            return output, state


        # embedding layer
        if self.config.use_embedding is False:
            self.embed_input = tf.one_hot(self.input_seqs, depth=self.vocab_size)  # 独热编码[1,2,3] depth=5 --> [[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]]，此时的输入节点个数为num_classes
        else:
            embedding = tf.get_variable('embedding', [self.vocab_size, self.config.embedding_dim])
            self.embed_input = tf.nn.embedding_lookup(embedding, self.input_seqs)  # 词嵌入[1,2,3] --> [[3,...,4],[0.7,...,-3],[6,...,9]],embeding[depth*embedding_size]=[[0.2,...,6],[3,...,4],[0.7,...,-3],[6,...,9],[8,...,-0.7]]，此时的输入节点个数为embedding_size
        self.embed_input = tf.nn.dropout(self.embed_input, keep_prob=self.keep_prob)


        # bilstm layer
        with tf.variable_scope("bi_lstm"):
            # 定义ann网络
            ann_output, ann_state = bilstm(self.config.hidden_dim, self.config.num_layers, self.embed_input, self.input_length)
            self.ann_output = tf.concat(ann_output, -1)  # fw,bw 输出拼接
            # layers_state = []
            # for layer_id in range(self.config.num_layers):
            #     layers_state.append(ann_state[0][layer_id])  # fw_layer_i:（c,h）
            #     layers_state.append(ann_state[1][layer_id])  # bw_layer_i:(c,h)
            # self.ann_state = tuple(layers_state)  # ((c,h),(c,h),(c,h),(c,h),...)

        # project layer
        with tf.variable_scope("project"):
            w_1 = tf.get_variable("w_1",[self.config.hidden_dim*2, self.config.hidden_dim],initializer= tf.random_normal_initializer(mean=0,stddev=0.1))  # +1是因为对未知的可能输出
            b_1 = tf.get_variable("b_1", [self.config.hidden_dim],initializer=tf.zeros_initializer())

            rep_output = tf.reshape(self.ann_output, [-1, self.config.hidden_dim*2])  # 序列展开
            fc1 = tf.tanh(tf.nn.xw_plus_b(rep_output, w_1, b_1))
            fc1_out = tf.nn.dropout(fc1, keep_prob=self.keep_prob)

            w_2 = tf.get_variable("w_2", [self.config.hidden_dim, self.config.num_classes],initializer=tf.random_normal_initializer(mean=0, stddev=0.1))  # +1是因为对未知的可能输出
            b_2 = tf.get_variable("b_2", [self.config.num_classes], initializer=tf.zeros_initializer())

            fc2 = tf.tanh(tf.nn.xw_plus_b(fc1_out, w_2, b_2))
            fc2_out = tf.nn.dropout(fc2, keep_prob=self.keep_prob)

            self.logits = tf.reshape(fc2_out, [-1, self.config.seq_length, self.config.num_classes], name="logits")

        with tf.name_scope("loss"):
            if not self.config.use_crf:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
                self.mean_loss = tf.reduce_mean(losses, name='cross_entropy_mean_loss')
            else:
                # crf layer
                log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=self.logits,
                    tag_indices=self.targets,
                    sequence_lengths=self.input_length )
                self.mean_loss = tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')


        with tf.name_scope("optimize"):
            # 优化器
            reg = tf.nn.l2_loss(w_1) + tf.nn.l2_loss(w_2) + tf.nn.l2_loss(b_1) + tf.nn.l2_loss(b_2)
            self.mean_loss += reg
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.mean_loss, global_step=self.global_step)

        with tf.name_scope("score"):
            self.y_cos = tf.nn.softmax(logits=self.logits, axis=-1)
            self.y_pre = tf.argmax(self.y_cos, -1)

    def load(self, checkpoint):
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))

    def train(self, batch_train_g, model_path, val_g):
        with self.session as sess:
            for q, r, q_len in batch_train_g:
                start = time.time()
                feed = {self.input_seqs: q,
                        self.input_length: q_len,
                        self.targets: r,
                        self.keep_prob: self.config.train_keep_prob}
                batch_loss, _ ,y_pre, y_cos,logits = sess.run([self.mean_loss, self.optim, self.y_pre, self.y_cos,self.logits], feed_dict=feed)
                end = time.time()

                # control the print lines
                if self.global_step.eval() % self.config.log_every_n == 0:
                    print('step: {}/{}... '.format(self.global_step.eval(), self.config.max_steps),
                          'loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end - start)))

                if (self.global_step.eval() % self.config.save_every_n == 0):
                    y_pres = np.array([])
                    y_coss = np.array([])
                    y_s = np.array([])
                    for q, r, q_len in val_g:
                        feed = {self.input_seqs: q,
                                self.input_length: q_len,
                                self.targets: r,
                                self.keep_prob: 1}

                        y_pre, y_cos,logits = sess.run([self.y_pre, self.y_cos,self.logits], feed_dict=feed)
                        print(y_pre)
                        y_pres = np.append(y_pres, y_pre)
                        y_coss = np.append(y_coss, y_cos)
                        y_s = np.append(y_s, r)

                    # 计算预测准确率
                    print('val len:',len(y_s))
                    print("accuracy:{:.2f}%.".format((y_s == y_pres).mean() * 100),
                            'best:{:.2f}%'.format(self.global_loss.eval()* 100))

                    acc_val = (y_s == y_pres).mean()
                    if acc_val > self.global_loss.eval():
                        print('save best model...')
                        update = tf.assign(self.global_loss, acc_val)  # 更新最优值
                        sess.run(update)
                        self.saver.save(sess, os.path.join(model_path, 'model'), global_step=self.global_step)

                if self.global_step.eval() >= self.config.max_steps:
                    break

    def test(self, a_sample):
        '''
        input:q,r
        :return:0 or 1 
        '''
        sess = self.session
        q,  r, q_len = a_sample
        feed = {self.input_seqs: np.reshape(q,[1,-1]),
                self.input_length: np.reshape(q_len,[-1]),
                self.targets: np.reshape(r,[1,-1]),
                self.keep_prob: 1}
        y_pre, y_cos = sess.run([self.y_pre, self.y_cos], feed_dict=feed)
        return y_pre[0],y_cos[0]

