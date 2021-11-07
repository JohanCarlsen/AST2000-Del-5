'''
EGEN KODE
eksempel på interpolering
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x = np.arange(1, 11)
y = 1 / x

f = interp1d(x, y)

x_new = np.arange(1, 10, 0.1)

y_new = f(x_new)

# plt.figure()
# plt.title('Plot of $f\,(x)=\\frac{1}{x}$ evaluated at $x\in[1,2,\dots,10]$')
# plt.plot(x, y, 'o')
# plt.xlabel('x')
# plt.ylabel('$f\,(x)$')
# # plt.savefig('uninterpolated-example.png')

plt.figure()
plt.title('Interpolate $f\,(x)=\\frac{1}{x}$')
plt.plot(x, y, 'o', x_new, y_new)
plt.xlabel('x')
plt.ylabel('$f\,(x)$')
# plt.savefig('interpolate-example.png')

plt.show()

# n = np.linspace(0.1, 0.9, 9)
# for i in range(len(n)):
#     x_test = np.arange(1,10,n[i])
#     y_test = f(x_test)
#     plt.plot(x_test, y_test, label=f'dx={n[i]:.1f}')
# plt.legend()
# plt.show()
