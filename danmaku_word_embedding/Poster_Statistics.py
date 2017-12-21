import re
import jieba
import jieba.posseg as pseg
import os
import matplotlib.pyplot as plt

class Poster_Statistics(object):
    def __init__(self):
        pass

    def  calc_poster(self,dir):
        self.global_poster={}
        self.local_poster_list=[]
        self.lines=0
        for file_name in os.listdir(dir):
            f=open(os.path.join(dir,file_name),"r")
            #print(file_name)
            local_poster={}
            for lineNo, line in enumerate(f.readlines()):
                pattern = re.compile("^<d p=\"(.+)\">(.+)</d>")
                m = pattern.match(line)
                if m:
                    temp = m.group(1).split(',')[-2]
                    #print(temp)
                    if temp not in local_poster:
                        local_poster[temp]=0
                    else:
                        local_poster[temp]+=1
                    if temp not in self.global_poster:
                       self.global_poster[temp]=0
                    else:
                       self.global_poster[temp]+=1
                    self.lines+=1
            self.local_poster_list.append(local_poster)

    def calc(self):
        print(self.lines)
        print("zeze")
        print(len(self.global_poster))
        print(max(list(self.global_poster.values())))
        print("haha")
        self.dot=[]
        self.dot2=[]
        self.dot3=None
        for index,item in enumerate(self.local_poster_list):
            print(index)
            print(len(item))
            #self.dot.append(len(item))
            print(max(list(item.values())))
            self.dot2.append(max(list(item.values())))
            if index==33:
                self.dot3=item.values()
            #print(list(item.keys()))


    def draw1(self):
        plt.figure('the number of poster for each episode')
        plt.plot(self.dot)
        plt.show()

    def draw2(self):
        plt.figure('the number of poster for each episode')
        plt.plot(self.dot2)
        plt.show()

    def draw3(self):
        plt.figure('the number of poster for each episode')
        temp=sorted(self.dot3,reverse=True)
        plt.plot(temp)
        plt.show()




if __name__ == '__main__':
    p=Poster_Statistics()
    p.calc_poster("/Users/yinfeng/PycharmProjects/Danmaku/data/danmu")
    p.calc()
    #p.draw()
    #p.draw2()
    p.draw3()