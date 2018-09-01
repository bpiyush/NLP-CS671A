import matplotlib.pyplot as plt
import numpy as np

x = [3.52, 5.31, 6.09, 6.41, 6.94, 7.35, 7.57, 7.57, 8.33, 8.62, 9.25, 9.61, 10.8, 10.87]
y = [0.6, 0.78, 0.88, 0.99, 1.13, 1.31, 1.42, 1.52, 1.78, 2.02, 2.10, 2.00, 1.53, 1.53 ]

x2 = [3.52, 5.20, 5.81, 6.24, 6.41, 6.75, 7.57, 7.57, 8.62, 8.92, 9.25, 9.61, 10.0, 10.4]
y2 = [0.65, 1.01, 1.31, 1.63, 1.96, 2.05, 2.44, 2.18, 1.38, 1.33, 1.26, 1.21, 0.98, 0.82]

x3 = [3.73, 4.80, 5.5, 6.24, 7.14, 7.81, 8.06, 8.33, 8.92, 9.25, 9.61, 10.8, 11.3]
y3 = []
for t in range(len(y2)):
	y2[t] /= 0.7

"""
plt.plot(x, y)
plt.show()"""

fig = plt.figure()
plt.plot(x2, y2)
fig.suptitle('Transmissibility vs Frequency (With Extra Mass with Damper)', fontsize=12)
plt.xlabel('Frequency (Hz)', fontsize=12)
plt.ylabel('Transmissibility', fontsize=12)
fig.savefig('graph2.jpg')
