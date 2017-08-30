from DBSCAN import *
import numpy as np
from gensim.models import word2vec

try:
    import cPickle as pickle
except ImportError:
    import pickle

def grab_word2vec_calc():
    fr = open("data/cache/word2vec_model", "rb")
    model = pickle.load(fr)
    fr.close()
    return model

class DanmakuModel(object):
    def __init__(self):
        self.comment_list=[]
        self.error = 0
        self._initiallize_comment_list()


    def _initiallize_comment_list(self,filename="data/danmu/1.xml"):
        word_2_vec=grab_word2vec_calc()
        lines, self.vocabulary_list, self.vocabulary_size, self.vocabulary = DataPreProcessing()._proxy_(file_name,
                                                                                                         POS_tag)
        self.N = len(lines)
        for index, line in enumerate(lines):
            # calc mean sentence vector by word2vec
            total = np.zeros(300)
            for item in line["text"]:
                if item in word_2_vec:
                    total += word_2_vec[item]
                else:
                    self.error+=1

            total= total / len(line["text"])
            self.comment_list.append((line["lineno"],total))
    #[[(0.719, 0.103), (0.556, 0.215), (0.481, 0.149), (0.666, 0.091), (0.639, 0.161), (0.748, 0.232),

    def main(self):
        C = DBSCAN(self.comment_list, 0.11, 5)
        for item in C:
           print(len(item))



if __name__ == '__main__':
    d=DakumuModel()
    d.main()
    #print(set(dataset))