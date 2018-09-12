import os
import numpy as np
import pickle
import random


def batch_generator(samples, batchsize):
    '''产生训练batch样本'''
    n_samples = len(samples)
    n_batches = int(n_samples/batchsize)
    n = n_batches * batchsize
    while True:
        random.shuffle(samples)  # 打乱顺序
        for i in range(0, n, batchsize):
            batch_samples = samples[i:i+batchsize]
            batch_q = []
            batch_q_len = []
            batch_r = []
            batch_r_len = []
            batch_y = []
            for sample in batch_samples:
                batch_q.append(sample[0])
                batch_q_len.append(sample[1])
                batch_r.append(sample[2])
                batch_r_len.append(sample[3])
                batch_y.append(sample[4])

            batch_q = np.array(batch_q)
            batch_q_len = np.array(batch_q_len)
            batch_r = np.array(batch_r)
            batch_r_len = np.array(batch_r_len)
            batch_y = np.array(batch_y)
            yield batch_q,batch_q_len,batch_r,batch_r_len,batch_y

def get_sens_tags(file):
    with open(file, 'r', encoding='utf8') as f:
        sens_tags = []
        sen_words = []
        sen_tags = []
        n_sens = 0
        for line in f:
            line = line.strip("\n")
            if line:
                data = line.split("\t")
                sen_words.append(data[0])
                sen_tags.append(data[1])
            elif sen_words:
                n_sens += 1
                sens_tags.append((sen_words,sen_tags))
                sen_words = []
                sen_tags = []
        print('句子数量：%s' % n_sens)
    return sens_tags


class TextConverter(object):
    def __init__(self, text_file=None, save_file=None, max_vocab=5000 ):
        if os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                self.tuple_pkl = pickle.load(f)
        else:
            self.read_text(text_file, max_vocab)
            self.save_to_file(save_file)

        self.word_to_int_table = self.tuple_pkl[0]
        self.int_to_word_table = self.tuple_pkl[1]
        self.tag_to_id_table = self.tuple_pkl[2]
        self.id_to_tag_table = self.tuple_pkl[3]

    @property
    def vocab_size(self):
        return len(self.word_to_int_table)+1

    @property
    def tag_size(self):
        return len(self.tag_to_id_table)

    def read_text(self, text_file, max_vocab):
        with open(text_file, 'r', encoding='utf8') as f:
            text = []
            tag_to_id = {}
            for line in f:
                line = line.strip("\n")
                if not line:
                    continue
                data = line.split("\t")
                text.append(data[0])
                tag = data[1]
                if tag not in tag_to_id:
                    tag_to_id[tag] = len(tag_to_id)

            vocab = set(text)
            print('字符数量：%s ' % len(vocab))
            # max_vocab_process
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            word_to_id = {c: i for i, c in enumerate(vocab)}
            id_to_word = {v: k for k, v in word_to_id.items()}
            id_to_tag = {v: k for k, v in tag_to_id.items()}
            self.tuple_pkl = (word_to_id, id_to_word, tag_to_id, id_to_tag)

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.word_to_int_table)

    def tag_to_int(self, tag):
        if tag in self.tag_to_id_table:
            return self.tag_to_id_table[tag]
        else:
            return len(self.tag_to_id_table)

    def int_to_tag(self, id):
        if id in self.id_to_tag_table:
            return self.id_to_tag_table[id]
        else:
            return 'uk'

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.tuple_pkl, f)

    def text_to_arr(self, sen, tags, seq_length):
        sen_arr = []
        tags_arr = []
        last_sen_num = len(self.word_to_int_table)
        last_tag_num = len(self.tag_to_id_table)
        assert len(sen)==len(tags),'error incorrect lens sens or tags!!'
        query_len = len(sen)
        for i in range(query_len):
            sen_arr.append(self.word_to_int(sen[i]))
            tags_arr.append(self.tag_to_int(tags[i]))

        # padding
        if query_len < seq_length:
            sen_arr += [last_sen_num] * (seq_length - query_len)
            tags_arr += [last_tag_num] * (seq_length - query_len)
        else:
            sen_arr = sen_arr[:seq_length]
            tags_arr = tags_arr[:seq_length]
            query_len = seq_length

        return np.array(sen_arr), np.array(tags_arr), np.array(query_len)

    def QAs_to_arr(self, QAs, seq_length):
        QA_arrs = []
        for query, response in QAs:
            # text to arr
            query_arr,response_arr,query_len = self.text_to_arr(query, response, seq_length)
            QA_arrs.append([query_arr,response_arr,query_len])
        return QA_arrs



    def batch_generator(self,QA_arrs, batchsize):
        '''产生训练batch样本'''
        n_samples = len(QA_arrs)
        n_batches = int(n_samples / batchsize)
        n = n_batches * batchsize
        while True:
            random.shuffle(QA_arrs)  # 打乱顺序
            for i in range(0, n, batchsize):
                batch_samples = QA_arrs[i:i + batchsize]
                batch_q = []
                batch_q_len = []
                batch_r = []
                for sample in batch_samples:
                    batch_q.append(sample[0])
                    batch_r.append(sample[1])
                    batch_q_len.append(sample[2])

                yield np.array(batch_q),  np.array(batch_r), np.array(batch_q_len),


    def val_samples_generator(self,QA_arrs, batchsize=500):
        '''产生验证样本，batchsize分批验证，减少运行内存'''

        val_g = []
        n = len(QA_arrs)
        for i in range(0, n, batchsize):
            batch_samples = QA_arrs[i:i + batchsize]
            batch_q = []
            batch_q_len = []
            batch_r = []

            for sample in batch_samples:
                batch_q.append(sample[0])
                batch_r.append(sample[1])
                batch_q_len.append(sample[2])
            val_g.append((np.array(batch_q), np.array(batch_r), np.array(batch_q_len)))
        return val_g




if __name__ == '__main__':
    pass