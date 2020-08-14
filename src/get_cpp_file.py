import numpy as np
import os

path = 'weights/exp7'

files = os.listdir(path)

w1 = np.load(os.path.join(path, files[1]))
w2 = np.load(os.path.join(path, files[0]))

print('w1 shape \n')
print(w1.shape)

print('w2 shape \n')
print(w2.shape)



num_osc = w1.shape[1]
num_h = w1.shape[0]
num_out = w2.shape[0]

w1_str_real = 'float w1_real[%d][%d] = {'%(num_h, num_osc)
for i in range(num_h-1):
    w1_str_real = w1_str_real+'{'
    for j in range(num_osc-1):
        w1_str_real = w1_str_real+'%.3f,'%(w1[i][j].real)
    w1_str_real = w1_str_real+'%.3f},\n\t'%(w1[i][num_osc-1].real)
w1_str_real = w1_str_real+'{'
for j in range(num_osc-1):
    w1_str_real = w1_str_real + '%.3f, '%(w1[num_h-1][j].real)
w1_str_real = w1_str_real+'%.3f}};\n'%(w1[i][num_osc-1].real)

#print(w1_str_real)

w1_str_imag = 'float w1_imag[%d][%d] = {'%(num_h, num_osc)
for i in range(num_h-1):
    w1_str_imag = w1_str_imag+'{'
    for j in range(num_osc-1):
        w1_str_imag = w1_str_imag+'%.3f,'%(w1[i][j].imag)
    w1_str_imag = w1_str_imag+'%.3f},\n\t'%(w1[i][num_osc-1].imag)
w1_str_imag = w1_str_imag+'{'
for j in range(num_osc-1):
    w1_str_imag = w1_str_imag + '%.3f, '%(w1[num_h-1][j].imag)
w1_str_imag = w1_str_imag+'%.3f}};\n'%(w1[i][num_osc-1].imag)

#print(w1_str_imag)

w2_str_real = 'float w2_real[%d][%d] = {'%(num_out, num_h)
for i in range(num_out-1):
    w2_str_real = w2_str_real+'{'
    for j in range(num_h-1):
        w2_str_real = w2_str_real+'%.3f,'%(w2[i][j].real)
    w2_str_real = w2_str_real+'%.3f},\n\t'%(w2[i][num_h-1].real)
w2_str_real = w2_str_real+'{'
for j in range(num_h-1):
    w2_str_real = w2_str_real + '%.3f,'%(w2[num_out-1][j].real)
w2_str_real = w2_str_real+'%.3f}};\n'%(w2[i][num_h-1].real)

#print(w2_str_real)

w2_str_imag = 'float w2_imag[%d][%d] = {'%(num_out, num_h)
for i in range(num_out-1):
    w2_str_imag = w2_str_imag+'{'
    for j in range(num_h-1):
        w2_str_imag = w2_str_imag+'%.3f,'%(w2[i][j].imag)
    w2_str_imag = w2_str_imag+'%.3f},\n\t'%(w2[i][num_h-1].imag)
w2_str_imag = w2_str_imag+'{'
for j in range(num_h-1):
    w2_str_imag = w2_str_imag + '%.3f, '%(w2[num_out-1][j].imag)
w2_str_imag = w2_str_imag+'%.3f}};\n'%(w2[i][num_h-1].imag)

#print(w2_str_imag)

file_str = w1_str_real + w1_str_imag + w2_str_real + w2_str_imag
print(file_str)

f = open(os.path.join(path,'data.h'), 'w')
f.write(file_str)
f.close
