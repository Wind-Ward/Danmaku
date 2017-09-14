import numpy as np
from os import listdir
def sigmoid(x):
    return 1/(1+np.exp(-x))


class logistic_regresion(object):
    def __init__(self,inputs_dim):
        self.dim=inputs_dim
        self.W=np.zeros((self.dim,1))
        self.b=0

    def calc(self,X,Y,m,rate,epoches):
        #forward
        for _ in range(epoches):
            Z=np.dot(self.W.T,X)+self.b
            A=sigmoid(Z)
            dw=np.dot(X,(A-Y).T)
            db=np.sum(A-Y)
            self.W-=rate*1/m*dw
            self.b-=rate*1/m*db

    def main(self,train_path,test_path,rate,epoches):
        train_data,train_label=self.loadData(train_path)
        self.calc(train_data,train_label,len(train_data),rate,epoches)

        #test
        test_data, test_label = self.loadData(test_path)
        self.forward(test_data,test_label,len(test_data))

    def forward(self,X,Y,m):
        A = sigmoid(np.dot(self.W.T, X) + self.b)
        loss=-np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))*1/m
        print("loss: "+str(loss))

    def loadData(self,direction):
        trainfileList = listdir(direction)
        m = len(trainfileList)
        dataArray = np.zeros((m, 1024))
        labelArray = np.zeros((m, 1))
        for i in range(m):
            returnArray = np.zeros((1, 1024))  # 每个txt文件形成的特征向量
            filename = trainfileList[i]
            fr = open('%s/%s' % (direction, filename))
            for j in range(32):
                lineStr = fr.readline()
                for k in range(32):
                    returnArray[0, 32 * j + k] = int(lineStr[k])
            dataArray[i, :] = returnArray  # 存储特征向量

            filename0 = filename.split('.')[0]
            label = filename0.split('_')[0]
            labelArray[i] = int(label)  # 存储类别
        return dataArray.T, labelArray.T

if __name__ == '__main__':
    lr=logistic_regresion(1024)
    lr.main("../data/lr_data/train","../data/lr_data/test/",0.01,100)

