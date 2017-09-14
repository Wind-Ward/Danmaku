import numpy as np
import copy
import load_mnist

def sigmoid(input):
    return 1.0 / (1.0 + np.exp(-1 * input))

def derivate_sigmoid(input):
    return sigmoid(input) * (1 - sigmoid(input))

class Neural_NetWork(object):
    def __init__(self,layer):
        self.layer = layer
        self.size = len(layer)
        self.initialize_W_b()

    def initialize_W_b(self):
        self.W = [np.random.randn(row, column) for row, column in zip(self.layer[1:], self.layer[:-1])]
        self.b = [np.random.randn(row, 1) for row in self.layer[1:]]

    def foreward(self, input):
        for w, b in zip(self.W, self.b):
            input = sigmoid(np.dot(w, input) + b)
        return input

    def back_prop(self,x,y):
        delta_W = [np.zeros([row, column]) for row, column in zip(self.layer[1:], self.layer[:-1])]
        delta_b = [np.zeros([row, 1]) for row in self.layer[1:]]

        _a=x
        a_list=[x]
        z_list=[]
        for w,b in zip(self.W,self.b):
            _z=np.dot(w,_a)+b
            z_list.append(_z)
            _a=sigmoid(_z)
            #print(_a.shape)
            a_list.append(_a)

        delta=(a_list[-1]-y)*derivate_sigmoid(z_list[-1])
        delta_b[-1]+=delta
        delta_W[-1]+=np.dot(delta,a_list[-2].T)
        for l in range(2,self.size):
            #print(delta_W[])
            _a=a_list[-l-1]
            delta=np.dot(self.W[-l+1].T,delta)*derivate_sigmoid(z_list[-l])
            delta_b[-l]+=delta
            delta_W[-l]+=np.dot(delta,_a.T)
        return delta_W,delta_b



    def update_delta(self,batch,rate):
        delta_W = [np.zeros([row, column]) for row, column in zip(self.layer[1:], self.layer[:-1])]
        delta_b = [np.zeros([row, 1]) for row in self.layer[1:]]
        for x,y in batch:
            _delta_W,_delta_b=self.back_prop(x,y)
            for index in range(self.size-1):
                delta_W[index]+=_delta_W[index]
                delta_b[index]+=_delta_b[index]
        for index in range(self.size-1):
            self.W[index]-=rate*(1.0/len(batch)*delta_W[index])
            self.b[index]-=rate*(1.0/len(batch)*delta_b[index])

    def SGD(self,train,epoch,batch_size,rate,test_data=None):
        for i in range(epoch):
            np.random.shuffle(train)
            train_list=[train[j:j+batch_size] for j in range(0,len(train),batch_size)]
            for batch in train_list:
                self.update_delta(batch,rate)
            if test_data:
                self.predict(test_data,i)

    def predict(self,data,index):
        result=0
        for x,y in data:
            _=np.argmax(self.foreward(x))
            if _==y:
                result+=1
        print("epoch"+str(index)+"  right: "+str(result)+" total:"+str(len(data)))


if __name__ == '__main__':
    # nn=Neural_NetWork([3,10,10,2])
    # train_data=[(np.random.randn(3,1),np.random.randint(0,2)) for item in range(100)]
    # test_data=[(np.random.randn(3,1),np.random.randint(0,2)) for item in range(5)]
    # nn.SGD(train_data,1000,2,0.5,test_data=test_data)
    training_data, validation_data, test_data = load_mnist.load_data_wrapper()
    net=Neural_NetWork([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

