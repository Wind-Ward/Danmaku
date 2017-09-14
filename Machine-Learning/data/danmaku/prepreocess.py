import numpy as np

def preprocess():
    f=open("danmaku_Ode_to_Joy _1.txt","r").readlines()
    with open("process_danmaku_Ode_to_Joy _1.txt","w") as f2:
        for item in f:
            s=item.strip().split(" ")
            temp=[]
            for i in s:
                if len(i)!=1:
                    temp.append(i)
            s2=" ".join(temp)
            f2.write(s2+"\n")



if __name__ == '__main__':
    preprocess()