import pywt
import math

x = [1,3,2,4]

h = [el*1/(4*math.sqrt(2)) for el in [1+math.sqrt(3), 3+math.sqrt(3), 3-math.sqrt(3), 1-math.sqrt(3)]]
g = [h[3],-h[2],h[1],-h[0]]

low_pass = h[3]*x[0]+h[2]*x[1]+h[1]*x[2]+h[0]*x[3]
high_pass = g[3]*x[0]+g[2]*x[1]+g[1]*x[2]+g[0]*x[3]

def low_pass_calc(x):
    return h[3]*x[0]+h[2]*x[1]+h[1]*x[2]+h[0]*x[3]

def high_pass_calc(x):
    return g[3]*x[0]+g[2]*x[1]+g[1]*x[2]+g[0]*x[3]

def calc_low_pass_over_array(x):
    return [low_pass_calc(x[i:i+4]) for i in range(0,len(x)-2,2)]

def calc_high_pass_over_array(x):
    return [high_pass_calc(x[i:i+4]) for i in range(0,len(x)-2,2)]

low_ans = calc_low_pass_over_array(x)
high_ans = calc_high_pass_over_array(x)


z = getWaveletCoefficients(x,'db2')
z = pywt.wavedec([el for el in range(1,33)], 'db2','per')
hsm = z[0]
hse = z[1:]
print(hse[0])
rootNode = low_pass_calc(hse[0])
print(rootNode)

# print(len(z[0]))
# print([len(l) for l in z[1:]])


