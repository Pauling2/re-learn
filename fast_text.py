#coding=utf8


import os

# 获取text_data文件夹下的所有文件路径
os.chdir(r'E:\lab_project\text_data\text_data')
print(os.getcwd())
temp_list = list(os.walk(r"."))
original = temp_list[0][0]
file_name = temp_list[0][2]
path_list = [original + "\\" + eve_name for eve_name in file_name]
print(temp_list)
print(path_list)
#创建所需文件
train_data = open(r"train_data.txt", "w", encoding="utf-8")
train_label = open(r"train_label.txt", "w", encoding="utf-8")
test_data = open(r"test_data.txt", "w", encoding="utf-8")
test_label = open(r"test_label.txt", "w", encoding="utf-8")
vocabulary = open(r"vocabulary.txt", "w", encoding="utf-8")

# 将原始数据进行标签分离与训练测试集分离
for every_path in path_list:
    with open(every_path, "r", encoding="utf-8") as temp_file:
        corpus = [eve for eve in temp_file if len(eve.strip("\n")) != 0]
        limit1 = len(corpus)*0.9
        limit2 = len(corpus)*0.1
        for i in range(len(corpus)):
            if limit2 < i < limit1:
                if corpus[i][:3] == "pos":
                    train_data.write(corpus[i][3:])
                    train_label.write("1" + "\n")
                else:
                    train_data.write(corpus[i][3:])
                    train_label.write("0" + "\n")
            else:
                if corpus[i][:3] == "pos":
                    test_data.write(corpus[i][3:])
                    test_label.write("1" + "\n")
                else:
                    test_data.write(corpus[i][3:])
                    test_label.write("0" + "\n")



# 创建字库vocabulary_2gram，包含原始数据中所有的字，写入vocabulary.txt待用
from nltk.util import ngrams
with open(r"test_data.txt", "r", encoding="utf-8") as file1:
    corpus1 = [eve for eve in file1]
print(len(corpus1))
with open(r"train_data.txt", "r", encoding="utf-8") as file2:
    corpus2 = [eve for eve in file2]
print(len(corpus2))
with open(r"vocabulary_2gram.txt","w",encoding="utf-8") as file3:
    word_list = []
    corpus = corpus1 + corpus2
    # print(len(corpus1),len(corpus2))
    print(len(corpus))
    for line in corpus:
        word_list.append([char for char in line])
    print(len(word_list))
    _2gramword_list = []
    for eve in word_list:
        temp = ngrams(eve,2)
        ##将ngram返回的迭代器中的turple，转换成两个字符长度的字符串
        for turple in temp:
            _2gramword_list.append(turple[0]+turple[1])
    print(temp,turple)
    # 使用list和set之间的转换达到去重的目的
    word_list = list(set(_2gramword_list))
    for word in word_list:
        file3.write(word + "\n")

print(len(word_list),len(corpus))




from nltk.util import ngrams
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.keras as kr
# import tensorflow as tf

with open(r"train_data.txt", "r", encoding="utf-8") as file1:
    corpus = [eve.strip("\n") for eve in file1]
    print(len(corpus))
with open(r"vocabulary_2gram.txt", "r", encoding="utf-8") as file2:
    vocabulary = [word.strip("\n") for word in file2]
with open(r"train_label.txt", "r", encoding="utf-8") as file3:
    label_list = [int(eve.strip("\n")) for eve in file3]
    print(len(label_list))
# assert len(label_list) == len(corpus)

word2id = {word:id_ for id_, word in enumerate(vocabulary)}
print(word2id)
def line2id_2gram(line):
    temp = []
    for char in line:
        temp.append(char)
    tep = [eve[0] + eve[1] for eve in ngrams(temp,2)]
    return [word2id[word] for word in tep]


train_list = [line2id_2gram(line) for line in corpus]
print(train_list)
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_list, 100)  # 长度一致train_x
train_y = tf.keras.utils.to_categorical(label_list, num_classes=2)
tf.reset_default_graph()
X_holder = tf.placeholder(tf.int32, [None, 100])  # 占位
Y_holder = tf.placeholder(tf.float32, [None, 2])


# 做词嵌入工作 注意71166是自由生成的行向量，这里是构建的vocabulary_2gram.txt中的大小
embedding = tf.get_variable('embedding', [71166, 60])  # 一种初始化变量的方法，随机初始化了矩阵变量
embedding_inputs = tf.nn.embedding_lookup(embedding, X_holder)  # lookup


# 神经网络结构 输入-取平均-softmax二分类器-输出
mean = tf.reduce_mean(embedding_inputs, axis=1)  # 将句子中的字按照字向量取平均值
logits = tf.layers.dense(mean, 2)  # 接一个60：2的softmax的分类器

learning_rate = tf.train.polynomial_decay(1e-2, 0, 1)  # rate = (rate - 0.0001) *(1 - 0 / 1) ^ (1) +0.0001

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder, logits=logits)
loss = tf.reduce_mean(cross_entropy)  # 熵的平均值
optimizer = tf.train.AdamOptimizer(learning_rate)  # 定义优化器
train = optimizer.minimize(loss)  # 将优化器与损失值连接起来

isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(logits, 1))  # 判断是否正确
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))  # 判断准确率

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

with open(r"test_data.txt", "r", encoding="utf-8") as file4:
    corpus_ = [eve.strip("\n") for eve in file4]
with open(r"test_label.txt", "r", encoding="utf-8") as file5:
    label_list_ = [int(eve.strip("\n")) for eve in file5]
assert len(label_list_) == len(corpus_)
test_list = [line2id_2gram(line) for line in corpus_]
test_x = kr.preprocessing.sequence.pad_sequences(test_list, 100)  # 长度一致train_x
test_y = kr.utils.to_categorical(label_list_, num_classes=2)

import random
for i in range(3000):
    selected_index = random.sample(list(range(len(train_y))), k=60)  # 批训练大小的意思就是多少个样本调整一次参数
    batch_X = train_x[selected_index]
    batch_Y = train_y[selected_index]
    session.run(train, {X_holder:batch_X, Y_holder:batch_Y})
    step = i + 1
    if step % 100 == 0:
        selected_index = random.sample(list(range(len(test_y))), k=150)
        batch_X = test_x[selected_index]
        batch_Y = test_y[selected_index]
        loss_value, accuracy_value = session.run([loss, accuracy], {X_holder:batch_X, Y_holder:batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))
