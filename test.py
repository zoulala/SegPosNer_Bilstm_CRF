'''

'''
import os
import tensorflow as tf
from read_utils import TextConverter
from model import Model,Config
from read_utils import get_sens_tags

def main(_):
    model_path = os.path.join('models', Config.file_name)

    vocab_file = os.path.join(model_path, 'vocab_tuples.pkl')

    # 获取测试问题
    sens_tags_test = get_sens_tags('data/test.txt')

    # 数据处理
    converter = TextConverter(None, vocab_file, max_vocab=Config.vocab_max_size)
    print('vocab size:',converter.vocab_size)

    # 产生测试样本
    test_QA_arrs = converter.QAs_to_arr(sens_tags_test, Config.seq_length)

    # 加载上一次保存的模型
    model = Model(Config, converter.vocab_size)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)


    # 测试
    print('start to testing...')
    n = len(test_QA_arrs)
    for i in range(n):
        y_pre, y_cos = model.test(test_QA_arrs[i])
        tags = [converter.int_to_tag(id) for id in y_pre[:test_QA_arrs[i][2]]]
        print('\nword / tag / pre')
        for j in range(test_QA_arrs[i][2]):
            print("{} / {} / {}".format(sens_tags_test[i][0][j],sens_tags_test[i][1][j],tags[j]))









if __name__ == '__main__':
    tf.app.run()
