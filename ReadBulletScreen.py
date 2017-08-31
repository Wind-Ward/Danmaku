

import re
import jieba
import jieba.posseg as pseg
import copy

from collections import OrderedDict

class BulletScreen(object):
    def __init__(self):
        self.stop_words= set([
                    " ","the","of","is","and","to","in","that","we","for",\
                    "an","are","by","be","as","on","with","can","if","from","which","you",
                    "it","this","then","at","have","all","not","one","has","or","that","什么","一个"
                    ])


    def load_stop_words(self,file="data/metadata/stopWords.txt"):
        f = open(file)
        content = f.read()
        words = content.split('\n')
        for w in words:
            self.stop_words.add(w.strip())




    def read(self,file_name,POS_tag):
        f = open(file_name, "r")
        tempLine=[]
        vocabulary = OrderedDict()
        jieba.load_userdict("data/metadata/user_dict.txt")
        for lineNo,line in enumerate(f.readlines()):
            pattern=re.compile("^<d p=\"(.+)\">(.+)</d>")
            m=pattern.match(line)
            if m:
                temp={"time":int(float(m.group(1).split(',')[0])), \
                                   "text":[word  for word,flag in pseg.cut(m.group(2))  \
                                           if word not in self.stop_words and flag not in \
                                           POS_tag ],
                                   "lineno":lineNo+1}


                #提取有效弹幕 有效弹幕为长度>0的弹幕
                if len(temp["text"])>=2:
                    #print(temp["text"])
                    tempLine.append(temp)
                    for item in temp["text"]:
                        if item not in vocabulary:
                            vocabulary[item]=0


        lines=sorted(tempLine, key= lambda e:(e.__getitem__('time')))
        # print(vocabulary)
        return lines,vocabulary


    def run(self,file_name,POS_tag):
        self.load_stop_words()
        return  self.read(file_name,POS_tag)



if __name__=="__main__":
    # 所要分析的弹幕文件
    file_name = "data/danmu/1.xml"
    # 采用词性过滤的方式来过滤对弹幕挖掘没有实际意义的词 具体可查 http://www.cnblogs.com/adienhsuan/p/5674033.html
    POS_tag = ["m", "w", "g", "c", "o", "p", "z", "q", "un", "e", "r", "x", "d", "t", "h", "k", "y", "u", "s", "uj",
               "ul",
               "r", "eng"]
    lines,vocabulary=BulletScreen().run(file_name,POS_tag)
    print(lines)
    #lines
    #[{'lineno': 2041, 'time': 0, 'text': ['小伙伴', '你们好']}, {'lineno': 2729, 'time': 0, 'text': ['伪装', '着看', '完']}, {'lineno': 4227, 'time': 0, 'text': ['僵尸', '极品']},



