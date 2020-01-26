import numpy as np
import matplotlib.pyplot as plt

for i in range(2500):
    j = 1
    plt.plot(j, j*j)
    j += 1
plt.savefig('foo.png')
