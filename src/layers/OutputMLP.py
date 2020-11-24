import numpy as np
import math
import tensorflow as tf

class OutputMLP(object):
    def __init__(self):# num_osc, num_h, num_out, init = 'random'):
        self.num_osc = 20
        self.num_h = 20
        self.num_out = 2
        self.init = 'random'
        self.sig = np.vectorize(self._sigmoid)
        self.relu = np.vectorize(self._relu)
        self.relugrad = np.vectorize(self._relugrad)
        
    def _random_mat(self, s1, s2):
        return np.random.random((s1, s2))
    
    def build(self, num_osc, num_h, num_out, a, init = 'random'):
        self.a = a
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.init = init
        if self.init == 'random':
            self.W1 = self._random_mat(self.num_h, self.num_osc) + 1j*self._random_mat(self.num_h, self.num_osc)
            self.W2 = self._random_mat(self.num_out, self.num_h) + 1j*self._random_mat(self.num_out, self.num_h)
        elif self.init == 'zeros':
            self.W1 = np.zeros(
                shape = (self.num_h, self.num_osc),
                dtype = np.complex128) 
            self.w2 = np.zeros(
                shape = (self.num_out, self.num_h),
                dtype = np.complex128)
        else:
            raise ValueError('initialization strategey can only be either random or zeros for {init}'.format(init = self.init))

    def _sigmoid(self, X):
        if math.isclose(X, 0.0, rel_tol = 1e-5, abs_tol = 1e-5):
            return 0.0
        return 1/(1+np.exp(-X))
    
    def sigmoidf(self, x):
        #return self.sig(x)
        return -1+2/(1+np.exp(-self.a*x))
    
    def _relu(self, x):
        if x<0:
            return 0
        return x

    def reluf(self, X):
        return self.relu(X) 

    def set_W1(self, w1):
        self.W1 = w1

    def set_W2(self, w2):
        self.W2 = w2

    def backprop_sigmoid(self, yr, Z, data, lr):
        dW2r = np.matmul(
            -1*np.multiply(
                (data.signal[:, 1:self.num_out+1].T-yr),
                np.multiply(
                    (1+yr),
                    (1-yr)
                )*self.a/2
            ), 
            self.xhr.T
        )
        dW2i = np.matmul(
            np.multiply(
                (data.signal[:, 1:self.num_out+1].T-yr),
                np.multiply(
                    (1+yr),
                    (1-yr)
                )*self.a/2
            ), 
            self.xhi.T
        )
        dW1r = np.matmul(
            -1*np.multiply(
                np.matmul(
                    self.W2.real.T, 
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        np.multiply(
                            (1+yr),
                            (1-yr)
                        )*self.a/2
                    )
                ),
                np.multiply(
                    (1+self.xhr),
                    (1-self.xhr)
                )*self.a/2
            ), 
            Z.real.T
        ) + np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.imag.T, 
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        np.multiply(
                            (1+yr),
                            (1-yr)
                        )*self.a/2 
                    )   
                ),  
                np.multiply(
                    (1+self.xhi),
                    (1-self.xhi)
                )*self.a/2 
            ),  
            Z.imag.T
        )
        dW1i = -1*np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.real.T,
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        np.multiply(
                            (1+yr),
                            (1-yr)
                        )*self.a/2
                    )
                ),
                np.multiply(
                    (1+self.xhr),
                    (1-self.xhr)
                )*self.a/2
            ),
            Z.imag.T
        ) + np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.imag.T,
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        np.multiply(
                            (1+yr),
                            (1-yr)
                        )*self.a/2
                    )
                ),
                np.multiply(
                    (1+self.xhi),
                    (1-self.xhi)
                )*self.a/2
            ),
            Z.real.T
        ) 
        W2r = self.W2.real - lr*dW2r
        #print('dW2r')
        #print(np.max(dW2r))
        #print(np.min(dW2r))
        W2i = self.W2.imag - lr*dW2i
        #print('dW2i')
        #print(np.max(dW2i))
        #print(np.min(dW2i))
        self.set_W2(W2r + 1j*W2i)
        W1r = self.W1.real - lr*dW1r
        #print('dW1r')
        #print(np.max(dW1r))
        #print(np.min(dW1r))
        W1i = self.W1.imag - lr*dW1i
        #print('dW1i')
        #print(np.max(dW1i))
        #print(np.min(dW1i))
        self.set_W1(W1r+1j*W1i)

    def _relugrad(self, x):
        if x>=0:
            return 1
        else:
            return 0
    
    def relugradf(self, x):
        return self.relugrad(x)
    
    def backprop_relu(self, yr, Z, data, lr):
        dW2r = -1*np.matmul(
            np.multiply(
                (data.signal[:, 1:self.num_out+1].T-yr),
                self.relugradf(yr)
            ),  
            self.xhr.T
        )   
        dW2i = np.matmul(
            np.multiply(
                (data.signal[:, 1:self.num_out+1].T-yr),
                self.relugradf(yr)
            ),  
            self.xhi.T
        )
        dW1r = -1*np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.real.T,
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        self.relugradf(yr)
                    )
                ),
                self.relugradf(self.xhr)
            ),
            Z.T.real
        ) + np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.imag.T,
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        self.relugradf(yr)
                    )
                ),
                self.relugradf(self.xhi)
            ),
            Z.T.imag
        )
        dW1i = -1*np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.T.real,
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        self.relugradf(yr)
                    )
                ),
                self.relugradf(self.xhi)
            ),
            Z.imag.T
        ) + np.matmul(
            np.multiply(
                np.matmul(
                    self.W2.T.imag,
                    np.multiply(
                        (data.signal[:, 1:self.num_out+1].T-yr),
                        self.relugradf(yr)
                    )
                ),
                self.relugradf(self.xhr)
            ),
            Z.real.T
        )
        W2r = self.W2.real - lr*dW2r
        print('dW2r')
        print(np.max(dW2r))
        print(np.min(dW2r))
        W2i = self.W2.imag - lr*dW2i
        print('dW2i')
        print(np.max(dW2i))
        print(np.min(dW2i))
        self.set_W2(W2r + 1j*W2i)
        W1r = self.W1.real - lr*dW1r
        print('dW1r')
        print(np.max(dW1r))
        print(np.min(dW1r))
        W1i = self.W1.imag - lr*dW1i
        print('dW1i')
        print(np.max(dW1i))
        print(np.min(dW1i))
        self.set_W1(W1r+1j*W1i)
    

    def __call__(self, Z, activation):
        """
            Z np.ndarray (num_osc, N) The fourier component signals of the output
        """  
        #print(Z) 
        #print('weights')
        #print(self.W1)
        self.nh = np.matmul(self.W1, Z)
        self.nhr = self.nh.real
        self.nhi = self.nh.imag
        self.xhr = activation(self.nhr)
        self.xhi = activation(self.nhi)
        self.xh = self.xhr+1j*self.xhi
        #print(self.xh)
        self.no = np.matmul(self.W2, self.xh)
        self.nor = self.no.real
        self.noi = self.no.imag
        #"""
        self.yr = activation(self.nor)
        self.yi = activation(self.noi)
        """
        self.yr = self.reluf(self.nor)
        self.yi = self.reluf(self.noi)
        #"""
        self.y = self.yr + 1j*self.yi
        return self.yr
       
