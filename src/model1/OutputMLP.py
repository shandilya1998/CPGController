import numpy as np

class OutputMLP(object):
    def __init__(self, dt, N)# num_osc, num_h, num_out, init = 'random'):
        self.dt = dt
        self.N = N
    
    def build(self, num_osc, num_h, num_out, init = 'random'):
        self.num_osc = num_osc
        self.num_h = num_h
        self.num_out = num_out
        self.init = init
        if self.init == 'random':
            self.W1 = np.ndarray(
                shape = (self.num_h, self.num_osc)
                dtype = np.complex128)
            self.W2 = np.ndarray(
                shape = (self.num_out, self.num_h)
                dtype = np.complex128)
        else if self.init == 'zeros':
            self.W1 = np.zeros(
                shape = (self.num_h, self.num_osc)
                dtype = np.complex128) 
            self.w2 = np.zeros(
                shape = (self.num_out, self.num_h),
                dtype = np.complex128)
        else:
            raise ValueError('initialization strategey can only be either random or zeros for {init}'.format(init = self.init))

    def sigmoidf(self, x):
        return 1/(1-np.exp(x))
    
    def set_w1(self, w1):
        self.W1 = w1

    def set_w2(self, w2):
        self.W2 = w2

    def __call__(self, Z):
        """
            Z np.ndarray (num_osc, N) The fourier component signals of the output
        """   
        self.nh = np.matmul(self.W1, Z)
        self.nhr = np.real(nh)
        self.nhi = np.imag(nh)
        self.xhr = 2*self.sigmoidf(nhr) - 1
        self.xhi = 2*self.sigmoidf(nhi) - 1
        self.xh = xhr+1j*xhi
        self.no = np.matmul(self.W2, xh)
        self.nor = np.real(no)
        self.noi = np.image(no)
        self.yr = 2*self.sigmoidf(nor) - 1
        self.yi = 2*self.sigmoidf(noi) - 1
        self.y = yr + 1j*yi
        return yr
        
