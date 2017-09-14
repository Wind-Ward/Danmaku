# http://blog.csdn.net/u012319493/article/details/52802302

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def derivate_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def derivate_tanh(x):
    return 1 - np.tanh(x) ** 2


def int2bin(x, dim=8):
    _ = bin(x)[2:]
    size = len(_)
    while size < dim:
        _ = str(0) + _
        size += 1
    return _


class LSTM(object):
    def __init__(self, layer=[2, 100, 1]):
        self.input_dim = layer[0]
        self.hidden_dim = layer[1]
        self.output_dim = layer[2]
        self.bin_dim = 8
        self.init_parameters()

    def init_parameters(self):
        # forget gate
        self.W_f_h = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.W_f_x = np.random.randn(self.hidden_dim, self.input_dim)
        self.b_f = np.random.randn(self.hidden_dim, 1)
        # input gate
        self.W_i_h = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.W_i_x = np.random.randn(self.hidden_dim, self.input_dim)
        self.b_i = np.random.randn(self.hidden_dim, 1)
        # control
        self.W_control_h = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.W_control_x = np.random.randn(self.hidden_dim, self.input_dim)
        self.b_control = np.random.randn(self.hidden_dim, 1)
        # output gate
        self.W_o_h = np.random.randn(self.hidden_dim, self.hidden_dim)
        self.W_o_x = np.random.randn(self.hidden_dim, self.input_dim)
        self.b_o = np.random.randn(self.hidden_dim, 1)

        # out
        self.W_y = np.random.randn(self.output_dim, self.hidden_dim)
        self.b_y = np.random.randn(self.output_dim, 1)



    def SGD(self, epoches, rate):

        for i in range(epoches):

            a = int(np.random.uniform(0, 256) / 2)
            b = int(np.random.uniform(0, 256) / 2)
            c = a + b

            a_bin = int2bin(a)
            b_bin = int2bin(b)
            x = {t: np.array([int(x[0]), int(x[1])]).reshape(2, 1) for t, x in enumerate(zip(reversed(a_bin), reversed(b_bin)))}

            c_bin = int2bin(c)
            y = {t: np.array(int(y)).reshape(1, 1) for t, y in enumerate(reversed(c_bin))}

            C = {}
            h = {}
            C[-1] = np.zeros((self.hidden_dim, 1))
            h[-1] = np.zeros((self.hidden_dim, 1))

            # forget gate
            f_in = {}
            f_out = {}

            # input gate
            i_in = {}
            i_out = {}

            # control
            control_in = {}
            control_out = {}

            # output gate
            o_in = {}
            o_out = {}

            y_in = {}
            y_out = {}

            loss = 0

            # forward
            result=""
            for t in range(self.bin_dim):
                # forget gate
                f_in[t] = np.dot(self.W_f_x, x[t]) + np.dot(self.W_f_h, h[t - 1]) + self.b_f
                f_out[t] = sigmoid(f_in[t])

                # input gate
                i_in[t] = np.dot(self.W_i_x, x[t]) + np.dot(self.W_i_h, h[t - 1]) + self.b_i
                i_out[t] = sigmoid(i_in[t])

                # control
                control_in[t] = np.dot(self.W_control_x, x[t]) + np.dot(self.W_control_h, h[t - 1]) + self.b_control
                control_out[t] = np.tanh(control_in[t])

                C[t] = f_out[t] * C[t - 1] + i_out[t] * control_out[t]

                # output gate
                o_in[t] = np.dot(self.W_o_x, x[t]) + np.dot(self.W_o_h, h[t - 1]) + self.b_o
                o_out[t] = sigmoid(o_in[t])

                h[t] = o_out[t] * np.tanh(C[t])

                y_in[t] = np.dot(self.W_y, h[t]) + self.b_y
                y_out[t] = sigmoid(y_in[t])


                if(y_out[t]>0.5):
                    result="1"+result
                else:
                    result="0"+result


                loss += 1 / 2 * np.sum((y[t] - y_out[t]) ** 2)

            if i%1000==0:
                print("loss: " + str(loss / self.bin_dim))
                print("a: "+str(a)+" b: "+str(b)+" c: "+str(c))
                print("result: "+str(int(result,2)))


            # delta init

            # forget gate
            delta_W_f_h = np.zeros_like(self.W_f_h)
            delta_W_f_x = np.zeros_like(self.W_f_x)
            delta_b_f = np.zeros_like(self.b_f)

            # input gate
            delta_W_i_h = np.zeros_like(self.W_i_h)
            delta_W_i_x = np.zeros_like(self.W_i_x)
            delta_b_i = np.zeros_like(self.b_i)

            # control
            delta_W_control_h = np.zeros_like(self.W_control_h)
            delta_W_control_x = np.zeros_like(self.W_control_x)
            delta_b_control = np.zeros_like(self.b_control)

            # output gate
            delta_W_o_h = np.zeros_like(self.W_o_h)
            delta_W_o_x = np.zeros_like(self.W_o_x)
            delta_b_o = np.zeros_like(self.b_o)

            delta_W_y = np.zeros_like(self.W_y)
            delta_b_y = np.zeros_like(self.b_y)

            # error init
            error_f = np.zeros_like(self.b_f)
            error_i = np.zeros_like(self.b_i)
            error_control = np.zeros_like(self.b_control)
            error_o = np.zeros_like(self.b_o)
            error_h = np.zeros([self.hidden_dim, 1])
            error_C = np.zeros([self.hidden_dim, 1])

            # future_error init
            future_error_f = np.zeros_like(error_f)
            future_error_i = np.zeros_like(error_i)
            future_error_control = np.zeros_like(error_control)
            future_error_o = np.zeros_like(error_o)

            future_error_C = np.zeros_like(error_C)

            f_out[self.bin_dim] = np.zeros((self.hidden_dim, 1))

            # backward
            for t in reversed(range(self.bin_dim)):
                _ = (y_out[t] - y[t]) * derivate_sigmoid(y_in[t])


                delta_h = np.dot(self.W_y.T, _) + np.dot(self.W_i_h.T,future_error_i) + \
                          np.dot(self.W_f_h.T, future_error_f) + np.dot(self.W_o_h.T, future_error_o) + \
                          np.dot(self.W_control_h.T, future_error_control)

                error_o = delta_h * np.tanh(C[t]) * derivate_sigmoid(o_in[t])

                error_C = delta_h * o_out[t] * derivate_tanh(C[t]) + future_error_C * f_out[t + 1]

                error_f = error_C * C[t - 1] * derivate_sigmoid(f_in[t])

                error_i = error_C * control_out[t] * derivate_sigmoid(i_in[t])

                error_control = error_C * i_out[t] * derivate_sigmoid(control_in[t])


                delta_W_y += np.dot(_, h[t].T)
                delta_b_y += _


                delta_W_i_x += np.dot(error_i, x[t].T)
                delta_b_i += error_i
                delta_W_i_h += np.dot(error_i, h[t - 1].T)


                delta_W_f_x += np.dot(error_f, x[t].T)
                delta_b_f += error_f
                delta_W_f_h += np.dot(error_f, h[t - 1].T)

                delta_W_o_x += np.dot(error_o, x[t].T)
                delta_b_o += error_o
                delta_W_o_h += np.dot(error_o, h[t - 1].T)


                delta_W_control_x += np.dot(error_control, x[t].T)
                delta_b_control += error_control
                delta_W_control_h += np.dot(error_control, h[t - 1].T)

                future_error_f = error_f
                future_error_o = error_o
                future_error_i = error_i
                future_error_control = error_control
                future_error_C = error_C


            self.W_f_h -= 1 / self.bin_dim * rate * delta_W_f_h
            self.W_f_x -= 1 / self.bin_dim * rate * delta_W_f_x
            self.b_f -= 1 / self.bin_dim * rate * delta_b_f


            self.W_i_h -= 1 / self.bin_dim * rate * delta_W_i_h
            self.W_i_x -= 1 / self.bin_dim * rate * delta_W_i_x
            self.b_i -= 1 / self.bin_dim * rate * delta_b_i


            self.W_control_h -= 1 / self.bin_dim * rate * delta_W_control_h
            self.W_control_x -= 1 / self.bin_dim * rate * delta_W_control_x
            self.b_control -= 1 / self.bin_dim * rate * delta_b_control

            self.W_o_h -= 1 / self.bin_dim * rate * delta_W_o_h
            self.W_o_x -= 1 / self.bin_dim * rate * delta_W_o_x
            self.b_o -= 1 / self.bin_dim * rate * delta_b_o

            self.W_y -= 1 / self.bin_dim * rate * delta_W_y
            self.b_y -= 1 / self.bin_dim * rate * delta_b_y


    def main(self):
        self.SGD(10000,0.1)
        self.test()

    def test(self):
        pass



if __name__ == '__main__':
    l=LSTM()
    l.main()