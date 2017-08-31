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
file_name="data/danmu/33.xml"
timeInterval = 200
time_length = 2581

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
            comments=[]
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
                comments.append(tuple(_total))
            self.comment_list.append(comments)

        # print("slice 0 size:")
        # print(len(self.comment_list))


    def print_result(self,C,index):
        raw = open(file_name, "r").readlines()
        with open("result.txt", "a") as f:
            f.write("slice:"+str(index)+"\n")
            for i, cluster in enumerate(C):
                f.write("\tcluster:"+str(i))
                for j, item in enumerate(cluster):
                    #print(item) print(item[0])
                    print(raw[item[0]-1])
                    f.write("\t\t"+raw[item[0]-1])


    # [[(0.719, 0.103), (0.556, 0.215), (0.481, 0.149), (0.666, 0.091), (0.639, 0.161), (0.748, 0.232),

    def main(self):
        for index,slice in enumerate(self.comment_list):
            C = DBSCAN(slice, 0.08, 3)
            print("total cluster size:" + str(len(C)))
            self.print_result(C,index)


if __name__ == '__main__':
    d=DanmakuModel()
    d.main()
    print("error")
    print(d.error)
    #print(set(dataset))