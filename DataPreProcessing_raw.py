# -*- coding: utf-8 -*-

import numpy as np
from ReadBulletScreen import BulletScreen
from collections import OrderedDict



class DataPreProcessing(object):
    def __init__(self):
        self.docSet=[]

    def addRestComment(self):
        doc=[]
        while (len(self.lines) != 0):

                doc.append(self.lines[0])
                self.lines.pop(0)
        self.docSet.append(doc)


    def sliceWithTime(self,timeInterval,file_name,time_length,POS_tag):
        self.lines,vocabulary=BulletScreen().run(file_name,POS_tag)
        preTime=0
        lastTime=preTime+timeInterval
        for index in range(int(time_length/timeInterval)):
            doc =[]
            while(len(self.lines)!=0):
                if self.lines[0]["time"] <=lastTime:
                    doc.append(self.lines[0])
                    self.lines.pop(0)
                else:
                    preTime=lastTime
                    lastTime=preTime+timeInterval
                    self.docSet.append(doc)
                    doc=[]
                    break

        self.addRestComment()
        #print len(self.docSet)
        #self.print_docSet(self.docSet)
        self.print_raw_comment(file_name)
        return self.docSet

    def print_raw_comment(self,file_name):
        # raw=open(file_name,"r").readlines()
        with open("data/processed_danmu/processed_raw_"+str(file_name.split("/")[-1].split(".")[0]+str(".txt")),"w") as f:
            for i,slice in enumerate(self.docSet):
                #f.write("slice"+str(i)+"\n")
                for item in slice:
                    f.write(" ".join(item["text"]) + "\n")
    #
    # def print_slice_file(self, file_name):
    #     for i,slice in enumerate(self.docSet):
    #         f=open("data/processed_danmu/slice_file/"+"slice_"+str(i)+".txt","w")
    #         for item in slice:
    #             f.write(" ".join(item["text"])+"\n")




#[{'text': ['伪装', '着看', '完'], 'time': 0, 'lineno': 2730}, {'text': ['欢乐颂', '取', '汁源', '诚信', '发'], 'time': 0, 'lineno': 5308}, {'text': ['全集', '私', '私', '威信', '来来来'],
if __name__=="__main__":
    #时间片大小、单位秒
    timeInterval = 300
    # 所要分析的弹幕文件
    file_name = "data/danmu/33.xml"
    time_length = 2581

    POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj",
               "ul",
               "r", "eng"]
    print (DataPreProcessing().sliceWithTime(timeInterval,file_name,time_length,POS_tag)[0])



