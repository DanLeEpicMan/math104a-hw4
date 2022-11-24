'''
Daniel Naylor
5094024
11/23/2022
'''

# --------------------------------------------------------------
#                           Question 3
# --------------------------------------------------------------

import numpy as np
from numpy.fft import fft
from math import exp, sin, cos, pi, e
N=8 # number of data points


input_data = [
    exp(sin(j * 2 * pi / N))
    for j in range(N)
]

c = fft(input_data)


# note that f_0 = 1

# discrete 'a' coefficients
a = {
    0: 2/N * sum(input_data[1:])
}
for k in range(1, N//2 + 1):
    a[k] = np.real((c[N-k] + c[k] - 2)/(N))

# discrete 'b' coefficients
b = {
    0: 0
}
for k in range(1, N//2 + 1):
    b[k] = np.real((c[N-k] - c[k])/(1j * N))

def P_prime(x: float, /):
    '''
    The derivative of P_8
    '''
    fourier_sum = 0
    for k in range(1, N//2):
        fourier_sum += k * b[k] * cos(k * x) - k * a[k] * sin(k * x)

    return fourier_sum - 2 * a[4] * sin(4 * x)

def f_prime(x: float, /):
    '''
    The derivative of e^(sin(x))
    '''
    return cos(x) * e**(sin(x))

for j in range(8):
    x_j = j * 2 * pi / N
    print(f'Error for x_j = {j} * 2pi / N: {P_prime(x_j) - f_prime(x_j)}')

# Error for x_j = 0 * 2pi / N: -0.004317911098587035
# Error for x_j = 1 * 2pi / N: 1.2105558808948307
# Error for x_j = 2 * 2pi / N: -0.5000000000000002
# Error for x_j = 3 * 2pi / N: 0.20365768147826468
# Error for x_j = 4 * 2pi / N: 0.004317911098587146
# Error for x_j = 5 * 2pi / N: -0.2098362525369506
# Error for x_j = 6 * 2pi / N: 0.49999999999999994
# Error for x_j = 7 * 2pi / N: -1.2043773098361448
