import numpy as np

class RNN(object):
    def __init__(self, filename="input.txt"):
        self.data = open(filename, "r").read()
        self.seq_length = 10
        self.chars = list(set(self.data))
        self.vocabulary_size = len(self.chars)
        self.hidden = 100
        self.char_to_ix = {ch: ix for ix, ch in enumerate(self.chars)}
        self.ix_to_char = {ix: ch for ix, ch in enumerate(self.chars)}

        self.U = np.random.randn(self.hidden, self.vocabulary_size)*0.01
        self.W = np.random.randn(self.hidden, self.hidden)*0.01
        self.V = np.random.randn(self.vocabulary_size, self.hidden)*0.01
        self.bh = np.zeros((self.hidden, 1))
        self.by = np.zeros((self.vocabulary_size, 1))

    def forward(self, inputs, targets):
        loss = 0
        hs = {}
        in_hs={}
        ys = {}
        ps = {}
        for t in range(len(inputs)):
            if t == 0:
                in_hs[t]=np.dot(self.U, inputs[t]) + self.bh
                hs[t] = np.tanh(in_hs[t])
            else:
                in_hs[t]=np.dot(self.U, inputs[t]) + self.bh + np.dot(self.W, hs[t - 1])
                hs[t] = np.tanh(in_hs[t])
            ys[t] = np.dot(self.V, hs[t]) + self.by

            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
            loss += -1 * np.log(ps[t][targets[t],0])

        return loss, hs, ys, ps

    def update(self, inputs, targets, hprev):
        delta_U = np.zeros_like(self.U)
        delta_W = np.zeros_like(self.W)
        delta_V = np.zeros_like(self.V)
        delta_bh = np.zeros_like(self.bh)
        delta_by = np.zeros_like(self.by)

        loss, hs, ys, ps = self.forward(inputs, targets)
        dhraw=None
        #dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            ps[t][targets[t]] -= 1
            dy=ps[t]


            delta_V += np.dot(dy,hs[t].T)
            delta_by+=dy


            if t == len(inputs) - 1:
                dhraw = np.dot(self.V.T,dy) * (1 - hs[t] ** 2)
            else:
                dhraw = (np.dot(self.V.T,dy) + np.dot(self.W.T,dhraw)) * (1 - hs[t] ** 2)
            # dh = np.dot(self.V.T, dy) + dhnext
            # dhraw = (1 - hs[t] * hs[t]) * dh
            delta_bh+=dhraw
            if t != 0:
                delta_W += np.dot(dhraw, hs[t - 1].T)
            else:
                delta_W += np.dot(dhraw, hprev.T)
            delta_U += np.dot(dhraw, inputs[t].T)
            #dhnext = np.dot(self.W.T, dhraw)

        return delta_U, delta_W, delta_V, delta_bh, delta_by, loss, hs[len(inputs)-1]

    def SGD(self, epoches, rate, seq_length=10):
        p = 0
        hprev=None
        smooth_loss = -np.log(1.0 / self.vocabulary_size) * seq_length
        for i in range(epoches):
            inputs = []
            targets = []
            if p+seq_length+1>=len(self.data) or i==0:
                p=0
                hprev = np.zeros((self.hidden, 1))

            for ch in self.data[p:p + seq_length]:
                temp = np.zeros((self.vocabulary_size,1))
                temp[self.char_to_ix[ch]] = 1
                inputs.append(temp)
            targets = [self.char_to_ix[ch] for ch in self.data[p + 1:p + seq_length + 1]]



            delta_U, delta_W, delta_V, delta_bh, delta_by, loss,hprev = self.update(inputs, targets,hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            self.U -= rate * (1 / len(inputs) * delta_U)
            self.W -= rate * (1 / len(inputs) * delta_W)
            self.V -= rate * (1 / len(inputs) * delta_V)
            self.bh -= rate * (1 / len(inputs) * delta_bh)
            self.by -= rate * (1 / len(inputs) * delta_by)
            print("epoch: " + str(i) + " loss: " + str(smooth_loss))
            p+=seq_length



if __name__ == '__main__':
    r = RNN()
    r.SGD(10000, 1)
