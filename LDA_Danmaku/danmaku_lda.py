import numpy as np
from collections import OrderedDict
from collections import Counter
from scipy.special import gamma
from datetime import datetime
K=5
C=5
L=3

top_N = 10


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
        self.id2word={}

class Danmaku_LDA(object):
    def __init__(self,dpre,iters=1000):
        self.dpre=dpre
        self.iters=iters
        self.K=K
        self.C=C
        self.L=L
        self.D=self.dpre.docs_count
        self.V=self.dpre.words_count
        self.top_N=top_N


        #init parameters
        self.n_d_c=np.zeros(self.C,dtype=np.int32)
        self.n_d_c_l=np.zeros((self.C,self.L),dtype=np.int32)
        self.n_d_c_l_k=np.zeros((self.C,self.L,self.K),dtype=np.int32)
        self.n_w_c_l_k_v=np.zeros((self.C,self.L,self.K,self.V),dtype=np.int32)


        self.danmaku_C = np.random.randint(self.C, size=self.D)
        self.danmaku_L=np.random.randint(self.L, size=self.D)
        self.danmaku_Z=np.random.randint(self.K, size=self.D)
        self.danmaku_C_L_Z_list=[]

        for index,(c,l,z) in enumerate(zip(self.danmaku_C,self.danmaku_L,self.danmaku_Z)):
            self.n_d_c[c]+=1
            self.n_d_c_l[c,l]+=1
            self.n_d_c_l_k[c,l,z]+=1
            for id in self.dpre.docs[index].words:
                self.n_w_c_l_k_v[c,l,z,id]+=1
            self.danmaku_C_L_Z_list.append((c,l,z))


        self.delta=0.1
        self.gamma=0.1
        self.alpha=0.1
        self.beta=np.zeros((self.C,self.L,self.K,self.V),dtype=np.float32)

        #0 positive 1 negative 2 neural
        self.beta[:,0,:,:]=0.005
        self.beta[:,1,:,:] = 0.005
        self.beta[:,2,:,:] = 0.99

        self.omega=np.zeros(self.C,np.float32)
        self.pi=np.zeros((self.C,self.L),np.float32)
        self.theta=np.zeros((self.C,self.L,self.K),np.float32)
        self.phi=np.zeros((self.C,self.L,self.K,self.V),np.float32)
        self.init_beta()


    def init_beta(self):
        positive_list=[]
        with open("./util/positive.txt","r") as f:
            positive_list.extend(f.read().split("\n"))
            num=0
            for positive in positive_list:
                positive=positive.strip()
                if positive in self.dpre.word2id:
                    num+=1
                    index=self.dpre.word2id[positive]
                    self.beta[:,0,:,index]=0.99
                    self.beta[:,1,:,index]=0.005
                    self.beta[:,2,:,index]=0.005
            print("positive_num:%d" % num)
        negative_list=[]
        with open("./util/negative.txt","r") as f:
            negative_list.extend(f.read().split("\n"))
            num = 0
            for negative in negative_list:
                negative=negative.strip()
                if negative in self.dpre.word2id:
                    num+=1
                    index=self.dpre.word2id[negative]
                    self.beta[:, 0, :, index] = 0.005
                    self.beta[:, 1, :, index] = 0.99
                    self.beta[:, 2, :, index] = 0.005
            print("negative_num:%d" % num)


    def sampling(self,i):
        _c,_l,_k=self.danmaku_C_L_Z_list[i]
        word_id_list=self.dpre.docs[i].words
        _word_count=Counter(word_id_list)
        N_d=self.dpre.docs[i].length
        self.p=np.zeros((self.C,self.L,self.K),np.float32)
        for c in range(self.C):
            for l in range(self.L):
                for k in range(self.K):
                    _result = 1.0
                    _temp=0.0
                    for v, counter in _word_count.items():
                        _result *= gamma(self.n_w_c_l_k_v[c, l, k, v] + self.beta[c, l, k, v]+ counter) / gamma(self.n_w_c_l_k_v[c, l, k, v]  + self.beta[c, l, k, v])
                        _temp+=self.n_w_c_l_k_v[c,l,k,v]+self.beta[c,l,k,v]
                    self.p[c,l,k]=(self.n_d_c[c]+self.delta)*(self.n_d_c_l[c,l]+self.gamma)/\
                                  (self.n_d_c[c]+self.L*self.gamma)*(self.n_d_c_l_k[c,l,k]+self.alpha)/\
                                  (self.n_d_c_l[c,l]+self.K*self.alpha)*gamma(_temp)/\
                                  gamma(_temp+N_d)
                    self.p[c,l,k]*=_result

        _cum_p=np.cumsum(self.p)

        u = np.random.uniform(0, _cum_p[-1])
        for index,item in enumerate(_cum_p):
            if item>u:
                break

        self.n_d_c[_c] += 1
        self.n_d_c_l[_c, _l] += 1
        self.n_d_c_l_k[_c, _l, _k] += 1

        _c=int(index/(self.L*self.K))
        _temp=index%(self.L*self.K)
        _l=int(_temp/self.K)
        _k=_temp%self.K
        self.danmaku_C_L_Z_list[i]=(_c,_l,_k)



    def estimation(self):
        for x in range(self.iters):
            t1=datetime.now()
            for i in range(self.D):
                self.sampling(i)
            t2=datetime.now()
            print("iter:%d time:%s" % (x,str(t2-t1)))
        self.calc_paramters()
        self.write()



    def calc_paramters(self):
        self.omega=(self.n_d_c+self.delta)/(self.D+self.C*self.delta)
        for c in range(self.C):
            self.pi[c]=(self.n_d_c_l[c,:]+self.gamma)/(self.n_d_c[c]+self.L*self.gamma)
            for l in range(self.L):
                self.theta[c,l]=(self.n_d_c_l_k[c,l,:]+self.alpha)/(self.n_d_c_l[c,l]+self.K*self.alpha)
                for k in range(self.K):
                    self.phi[c,l,k]=(self.n_w_c_l_k_v[c,l,k]+self.beta[c,l,k])/(np.sum(self.n_w_c_l_k_v[c,l,k])+np.sum(self.beta[c,l,k]))


    def write(self):
        with open("character.txt","w") as f:
            f.write(str(self.omega))


        with open("sentiment_label.txt","w") as f:
            f.write(str(self.pi))

        with open("topic.txt","w") as f:
            f.write(str(self.phi))

        with open("character_sentimental_topic_analysis.txt","w") as f:
            for c in range(self.C):
                f.write("C%d:\n" % c)
                for l in range(self.L):
                    f.write("\t L%d:\n" % l)
                    for k in range(self.K):
                        id_list=np.argsort(-1*self.n_w_c_l_k_v[c,l,k])[:self.top_N]
                        f.write("\t")
                        for item in id_list:
                            f.write("\t%s" % self.dpre.id2word[item])
                        f.write("\n")


def preprocessing(trainfile):
    with open(trainfile, 'r') as f:
        docs = f.readlines()

    dpre = DataPreProcessing()
    items_idx = 0
    for line in docs:
        if line != "":
            tmp = line.strip().split(" ")
            doc = Document()
            for item in tmp:
                if item in dpre.word2id:
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
    dpre.id2word={v:k for k,v in dpre.word2id.items()}
    return dpre


def main(trainfile):
    dpre = preprocessing(trainfile)
    lda = Danmaku_LDA(dpre)
    s1=datetime.now()
    lda.estimation()
    s2=datetime.now()
    print("totoal time:%s" % str(s2-s1))

if __name__ == '__main__':
    main("./33.txt")

















