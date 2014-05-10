#!/usr/bin/python
# coding: utf-8

# MNISTを用いたニューラルネットによる手書き数字認識のデモコードです
# 学習方法やパラメータによりますが、だいたい 90 ~ 97% くらいの精度出ます。
# 使い方は、コードを読むか、
# python mnist_net.py -h
# としてください
# 参考までに、
# python mnist_net.py --epoches 50000 --learning_rate 0.1 --hidden 100
# とすると、テストセットに対して、93.2%の正解率です
# 僕の環境では、学習、認識合わせてた（だいたい）5分くらいかかりました。
# 201/05/10 Ryuichi Yamamoto

import numpy as np
from sklearn.externals import joblib
import cPickle
import gzip
import os

# 作成したニューラルネットのパッケージ
import net

def load_mnist_dataset(dataset):
    """
    MNISTのデータセットをダウンロードします
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

def augument_labels(labels, order):
    """
    1次元のラベルデータを、ラベルの種類数(order)次元に拡張します
    """
    new_labels = []
    for i in range(labels.shape[0]):
        v = np.zeros(order)
        v[labels[i]] = 1
        new_labels.append(v)
    
    return np.array(new_labels).reshape((labels.shape[0], order))        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MNIST手書き数字認識のデモ")
    parser.add_argument("--epoches", dest="epoches", type=int, required=True)
    parser.add_argument("--learning_rate", dest="learning_rate",\
                        type=float, default=0.1)
    parser.add_argument("--hidden", dest="hidden", type=int, default=100)
    args = parser.parse_args()

    train_set, valid_set, test_set = load_mnist_dataset("mnist.pkl.gz")
    n_labels = 10 # 0,1,2,3,4,5,6,7,9
    n_features = 28*28

    # モデルを新しく作る
    nn = net.NeuralNet(n_features, args.hidden, n_labels)

    # モデルを読み込む
    # nn = joblib.load("./nn_mnist.pkl")

    nn.train(train_set[0], augument_labels(train_set[1], n_labels),\
             args.epoches, args.learning_rate, monitor_period=2000)

    ## テスト
    test_data, labels = test_set
    results = np.arange(len(test_data), dtype=np.int)
    for n in range(len(test_data)):
        results[n] = nn.predict(test_data[n])
        # print "%d : predicted %s, expected %s" % (n, results[n], labels[n])
    print "recognition rate: ", (results == labels).mean()
    
    # モデルを保存
    model_filename = "nn_mnist.pkl"
    joblib.dump(nn, model_filename, compress=9)
    print "The model parameters are dumped to " + model_filename
