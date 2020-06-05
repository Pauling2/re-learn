#coding=utf8


from gensim.models.doc2vec import Doc2Vec,LabeledSentence

with open(r'E:\uniprot_sprot_oneline.txt')as f:
    lst1=f.readlines()
    lst2=[i.strip('\n') for i in lst1]
    f.close()
print len(lst2)
print lst2[1]

lst=[]
for i in lst2[:100000]:
    lst.append(i.split(' '))
#print lst

model=Doc2Vec(dm=0, dbow_words=1, size=100, window=8, min_count=10, iter=10, workers=4)

#SentimentDocument = namedtuple('SentimentDocument', 'words tags')

def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

lst_dic=labelizeReviews(lst,'train')

#x=x_train+x_test

model.build_vocab(lst_dic)
model.train(lst_dic, total_examples=model.corpus_count, epochs=model.iter)
model.save('doc.bin')
print model.infer_vector(["ASA",'SAD','DSF','SFT','FTS','TSS','SSP','SPE','PER','ERG','RGH','GHL'])



##github上训练wiki百科文本

import gensim.models as gm
import logging

#doc2vec参数
vector_size=256  #default 100
window_size=15  #窗口大小，表示当前词与预测词在一个句子中的最大距离
min_count=1    #词频少于min_count的单词会被丢弃掉,default 5
sampling_threshold=1e-5  #高频词的随机降采样的阈值,default 1e-3
negative_size=5  #>0时，采用negativesampling，用于设置noise words的个数（一般是5-20）
train_epcho=100  #迭代次数
dm=0  #0=dbow; 1=dmpv
worker_count=4  #用于控制训练的并行个数


#pretrained word embeddings
pretrained_emb='toy_data/pretrained_word_embeddings.txt'  #None if use without pretrained embeddings


#输入语料库
train_corpus='toy_data/wiki_en.txt'


#模型输出
saved_path='toy_data/model/wiki_en_doc2vec.model.bin'


#获取日志信息
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)


#训练doc2vec模型
docs=gm.doc2vec.TaggedLineDocument(train_corpus)
model=gm.Doc2Vec(docs,size=vector_size,window=window_size,min_count=min_count,sample=sampling_threshold,workers=worker_count,
                 hs=0,dm=dm,negative=negative_size,dbow_words=1,dm_concat=1,pretrained_emb=pretrained_emb,iter=train_epcho)
model.save(saved_path)


##模型的加载与使用

#python example to infer document vectors from trained doc2vec model
import gensim.models as gm
import codecs
import numpy as np

#parameters
model = "toy_data/model/wiki_en_doc2vec.model.bin"
test_docs = "toy_data/test.txt" # test.txt为需要向量化的文本
output_file = "toy_data/test_vector.txt" #得到测试文本的每一行的向量表示

# 超参
start_alpha = 0.01
infer_epoch = 1000

#加载模型

m = gm.Doc2Vec.load(model)
test_docs = [x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines()]

#infer test vectors

output = open(output_file, "w")

for d in test_docs:
    output.write(" ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n")
output.flush()
output.close()
#print(len(test_docs)) #测试文本的行数
print(m.most_similar("party", topn=5)) # 找到与party单词最相近的前5个

#保存为numpy形式
test_vector = np.loadtxt('toy_data/test_vector.txt')
test_vector = np.save('toy_data/test_vector', test_vector)