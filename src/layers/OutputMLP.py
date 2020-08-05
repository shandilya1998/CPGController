import numpy as np
import math

class OutputMLP(object):
    def __init__(self):# num_osc, num_h, num_out, init = 'random'):
        self.num_osc = 20
        self.num_h = 20
        self.num_out = 2
        self.init = 'random'
        self.sig = np.vectorize(self._sigmoid)
    
    def build(self, num_osc, num_h, num_out, init = 'random'):
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.init = init
        if self.init == 'random':
            self.W1 = np.empty(
                shape = (self.num_h, self.num_osc),
                dtype = np.complex128)
            self.W2 = np.empty(
                shape = (self.num_out, self.num_h),
                dtype = np.complex128)
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
            return 0
        else:
            return 1/(1-np.exp(-X))
    
    def sigmoidf(self, x):
        return self.sig(x) 
    
    def set_W1(self, w1):
        self.W1 = w1

    def set_W2(self, w2):
        self.W2 = w2

    def __call__(self, Z):
        """
            Z np.ndarray (num_osc, N) The fourier component signals of the output
        """   
        self.nh = np.matmul(self.W1, Z)
        self.nhr = self.nh.real
        self.nhi = self.nh.imag
        self.xhr = self.sigmoidf(self.nhr)
        self.xhi = self.sigmoidf(self.nhi)
        self.xh = self.xhr+1j*self.xhi
        self.no = np.matmul(self.W2, self.xh)
        self.nor = self.no.real
        self.noi = self.no.imag
        self.yr = self.sigmoidf(self.nor)
        self.yi = self.sigmoidf(self.noi)
        self.y = self.yr + 1j*self.yi
        return self.yr
       
