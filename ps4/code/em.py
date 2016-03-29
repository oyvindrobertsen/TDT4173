from __future__ import division

from math import exp, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn


def pdf_n(x, mean, std_dev):
    return (exp(-((x - mean)**2/(2 * (std_dev**2))))) / \
            (std_dev * sqrt(2 * pi))


def em_step(data, old_expectancy):
    # E
    E_A = []
    E_B = []
    for sample in data:
        E_A.append(pdf_n(sample, old_expectancy[0], 1) /
                (pdf_n(sample, old_expectancy[0], 1) +
                    pdf_n(sample, old_expectancy[1], 1)))
        E_B.append(pdf_n(sample, old_expectancy[1], 1) /
                (pdf_n(sample, old_expectancy[0], 1) +
                    pdf_n(sample, old_expectancy[1], 1)))
    # M
    new_A = sum(map(lambda (z, x): z * x, zip(E_A, data))) / sum(E_A)
    new_B = sum(map(lambda (z, x): z * x, zip(E_B, data))) / sum(E_B)
    return [new_A, new_B]

i = 0
MAX_ITER = 10000
TOL = 1e-7
expectancy = [-0.5, 0.5]
expectancies = [expectancy]
with open('sample-data.txt') as f:
    data = np.array(map(float, f.read().splitlines()))

while i < MAX_ITER:
    new_expectancy = em_step(data, expectancy)
    expectancies.append(new_expectancy)
    delta_A = abs(new_expectancy[0] - expectancy[0])
    delta_B = abs(new_expectancy[1] - expectancy[1])
    if delta_A < TOL and delta_B < TOL:
        break
    else:
        expectancy = new_expectancy
        i += 1

print expectancy, i
print expectancies
plt.title("Source data histogram and EM-fitted Gaussians")
plt.hist(data, bins=30, normed=True)
xs = np.linspace(-3, 6, 200)
plt.plot(xs, mlab.normpdf(xs, expectancy[0], 1))
plt.plot(xs, mlab.normpdf(xs, expectancy[1], 1))
plt.show()
