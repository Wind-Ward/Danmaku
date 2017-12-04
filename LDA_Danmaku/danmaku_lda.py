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
    def __init__(self):
        # global K
        # global C
        # global L
        self.K=K
        self.C=C
        self.L=L
        self.D=D


        #init parameters
        self.n_d_c=np.zeros(self.C,dtype=np.int32)
        self.n_d_c_l=np.zeros((self.C,self.L),dtype=np.int32)
        self.n_d_c_l_k=np.zeros((self.C,self.L,self.K),dtype=np.int32)
        self.n_w_c_l_k_v=np.zeros((self.C,self.L,self.K,self.V),dtype=np.int32)


        #self.danmaku_C=np.zeros(self.D,dtype=np.int32)
        self.danmaku_C = np.random.randint(self.C, size=self.D)
        self.danmaku_L=np.random.randint(self.L, size=self.D)
        self.danmaku_Z=np.random.randint(self.K, size=self.D)















