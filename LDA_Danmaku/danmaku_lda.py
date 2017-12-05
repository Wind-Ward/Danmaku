import numpy as np

K = 10
C=5
L=3
alpha = 0.1
beta = 0.1
iter_times = 1000
top_words_num = 10


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


class DataPreProcessing(object):
    def __init__(self):
        self.docs_count = 0
        self.words_count = 0
        self.docs = []
        self.word2id = OrderedDict()

class Danmaku_LDA(object):
    def __init__(self,dpre,iters=1000):
        self.dpre=dpre
        self.iters=iters
        self.K=K
        self.C=C
        self.L=L
        self.D=self.dpre.docs_count
        self.V=self.dpre.words_count


        #init parameters
        self.n_d_c=np.zeros(self.C,dtype=np.int32)
        self.n_d_c_l=np.zeros((self.C,self.L),dtype=np.int32)
        self.n_d_c_l_k=np.zeros((self.C,self.L,self.K),dtype=np.int32)
        self.n_w_c_l_k_v=np.zeros((self.C,self.L,self.K,self.V),dtype=np.int32)


        self.danmaku_C = np.random.randint(self.C, size=self.D)
        self.danmaku_L=np.random.randint(self.L, size=self.D)
        self.danmaku_Z=np.random.randint(self.K, size=self.D)

        for index,(c,l,z) in enumerate(zip(self.danmaku_C,self.danmaku_L,self.danmaku_Z)):
            self.n_d_c[c]+=1
            self.n_d_c_l[c,l]+=1
            self.n_d_c_l_k[c,l,z]+=1
            for id in self.dpre.docs[index].words:
                self.n_w_c_l_k_v[c,l,z,id]+=1

        self.delta=np.full(self.C,0.1,dtype=np.float32)
        self.gamma=np.full((self.C,self.L),0.1,dtype=np.float32)
        self.alpha=np.full((self.C,self.L,self.K),0.1,dtype=np.float32)
        self.beta=np.full((self.C,self.L,self.K,self.V),0.1,dtype=np.float32)


    def sampling(self,i):
        pass




    def estimation(self):
        for x in range(self.iters):
            for i in range(self.D):
                self.sampling(i)


def preprocessing(trainfile):
    with codecs.open(trainfile, 'r', 'utf-8') as f:
        docs = f.readlines()

    dpre = DataPreProcessing()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split(" ")
            # 生成一个文档对象
            doc = Document()
            for item in tmp:
                if dpre.word2id.has_key(item):
                    doc.words.append(dpre.word2id[item])
                else:
                    dpre.word2id[item] = items_idx
                    doc.words.append(items_idx)
                    items_idx += 1
            doc.length = len(tmp)
            dpre.docs.append(doc)
        else:
            pass
    dpre.docs_count = len(dpre.docs)
    dpre.words_count = len(dpre.word2id)
    return dpre


def main(trainfile):
    dpre = preprocessing(trainfile)
    lda = Danmaku_LDA(dpre.docs_count,dpre.words_count,trainfile)
    return lda.est()

if __name__ == '__main__':
    pass

















