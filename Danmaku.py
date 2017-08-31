from DBSCAN import *
import numpy as np
from gensim.models import word2vec
from DataPreProcessing import DataPreProcessing

try:
    import cPickle as pickle
except ImportError:
    import pickle

def grab_word2vec_calc():
    fr = open("data/cache/word2vec_model", "rb")
    model = pickle.load(fr)
    fr.close()
    return model


POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj",
               "ul","r", "eng"]
file_name="data/danmu/1.xml"
timeInterval = 200
time_length = 2582

class DanmakuModel(object):
    def __init__(self):
        self.comment_list=[]
        self.error = 0
        self._initiallize_comment_list()


    def _initiallize_comment_list(self):
        word_2_vec=grab_word2vec_calc()
        lines = DataPreProcessing().sliceWithTime(timeInterval,file_name,time_length,POS_tag)
        self.N = len(lines)
        for i,slice  in enumerate(lines):
            if i==1:
                break
            for index, line in enumerate(slice):
                total = np.zeros(300)
                print(line["text"])
                for item in line["text"]:
                    if item in word_2_vec:
                        total += word_2_vec[item]
                    else:
                        self.error+=1
                total= total / len(line["text"])
                _total=list(total)
                _total.insert(0, line["lineno"])
                self.comment_list.append(tuple(_total))
        print(self.comment_list[0])
        print("slice 0 size:")
        print(len(self.comment_list))

    #[[(0.719, 0.103), (0.556, 0.215), (0.481, 0.149), (0.666, 0.091), (0.639, 0.161), (0.748, 0.232),
    def main(self):
        C = DBSCAN(self.comment_list,0.1,5)
        print()
        #print(C)
        print("len C:"+str(len(C)))

        for item in C:
           print(len(item))



if __name__ == '__main__':
    d=DanmakuModel()
    d.main()
    print("error")
    print(d.error)
    #print(set(dataset))