from collections import OrderedDict

import numpy as np
from DBSCAN import *

from danmaku_word_embedding.DataPreProcessing_slice_file import DataPreProcessing

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
        self.keyword_result=[]


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
                _total.insert(0," ".join(line["text"]))
                _total.insert(0,line["content"])
                _total.insert(0, int(line["time"]))
                _total.insert(0,line["lineno"])
                comments.append(tuple(_total))
            self.comment_list.append(comments)

        # print("slice 0 size:")
        # print(len(self.comment_list))


    # def print_result(self,C,index):
    #     raw = open(file_name, "r").readlines()
    #     with open("result_33_0.005_2.txt", "a") as f:
    #         f.write("slice:"+str(index)+"\n")
    #         for i, cluster in enumerate(C):
    #             f.write("\tcluster:"+str(i)+"\n")
    #             for j, item in enumerate(cluster):
    #                 #print(item) print(item[0])
    #                 print(raw[item[0]-1])
    #                 f.write("\t\t"+raw[item[0]-1])

    # def print_result_2(self, C, index):
    #     with open("result_33_0.005_2_cluster_raw.txt", "a") as f:
    #         f.write("slice:" + str(index) + "\n")
    #         for i, cluster in enumerate(C):
    #             f.write("\tcluster:" + str(i) + "\n")
    #             for j, item in enumerate(cluster):
    #                 f.write("\t\t" +item[1]+"\n")
    #
    # def print_result_3(self, C, index):
    #     with open("./data/processed_danmu/slice_cluster_file/"+"result_33_0.005_2_cluster_slice_"+str(index)+".txt", "w") as f:
    #         for i, cluster in enumerate(C):
    #             for j, item in enumerate(cluster):
    #                 f.write(item[1]+"\n")


    # [[(0.719, 0.103), (0.556, 0.215), (0.481, 0.149), (0.666, 0.091), (0.639, 0.161), (0.748, 0.232),

    #[[(5615, 101, '小曲是为了让樊姐知道她老娘要借钱', '小曲 樊姐 老娘 借钱',
    def keyword(self,C):
        dict=OrderedDict()
        time=OrderedDict()
        for i,cluster in enumerate(C):
            for j,item in enumerate(cluster):
                for _ in item[3].split(" "):
                    if _ not in dict:
                        dict[_]=0
                    else:
                        dict[_]+=1
                    if _ not in time:
                        time[_]={"min":item[1],"max":item[1]}
                    else:
                        if item[1]>=time[_]["max"]:
                            time[_]["max"]=item[1]
                        elif item[1]<time[_]["min"]:
                            time[_]["min"]=item[1]

        tf=np.array(list(dict.values()))
        tf=tf/np.sum(tf)
        substract_time=[]
        for k,v in time.items():
            substract_time.append(v["max"]-v["min"])

        max_index=np.argsort(-np.log(np.array(substract_time))*tf)[:3]
        result=[]
        for i in max_index:
            result.append(list(dict.keys())[i])
        self.keyword_result.append(result)

    def print_keyword_result(self):
        with open("result_33_0.005_2_keyword.txt", "w") as f:
               for index,i in enumerate(self.keyword_result):
                   f.write("slice:"+str(index)+"\n")
                   for j in i:
                       f.write(("\t"+j)+"\n")


    def main(self):
        #print(self.comment_list)
        for index,slice in enumerate(self.comment_list):
            C = DBSCAN(slice,0.005,2)
            print("total cluster size:" + str(len(C)))
            self.keyword(C)
        self.print_keyword_result()

            #self.print_result(C,index)
            #self.print_result_2(C, index)
            #self.print_result_3(C,index)


if __name__ == '__main__':
    d=DanmakuModel()
    d.main()
    print("error:")
    print(d.error)
    #print(set(dataset))