import os
import tensorflow as tf
from read_utils import TextConverter
from model import Model,Config
from read_utils import get_sens_tags

def main(_):

    model_path = os.path.join('models', Config.file_name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    train_file = 'data/train.txt'
    save_file = os.path.join(model_path, 'vocab_tuples.pkl')

    # 获取样本数据
    sens_tags_train = get_sens_tags(train_file)
    sens_tags_val = get_sens_tags('data/dev.txt')


    # 数据处理
    converter = TextConverter(train_file, save_file, max_vocab=Config.vocab_max_size)
    print('vocab size:',converter.vocab_size)
    Config.num_classes = converter.tag_size+1

    # 产生训练样本
    train_QA_arrs = converter.QAs_to_arr(sens_tags_train, Config.seq_length)
    train_g = converter.batch_generator(train_QA_arrs, Config.batch_size)

    # 产生验证样本
    val_QA_arrs = converter.QAs_to_arr(sens_tags_val, Config.seq_length)
    val_g = converter.val_samples_generator(val_QA_arrs, Config.batch_size)

    # 加载上一次保存的模型
    model = Model(Config,converter.vocab_size)
    checkpoint_path = tf.train.latest_checkpoint(model_path)
    if checkpoint_path:
        model.load(checkpoint_path)

    print('start to training...')
    model.train(train_g, model_path, val_g)



if __name__ == '__main__':
    tf.app.run()
