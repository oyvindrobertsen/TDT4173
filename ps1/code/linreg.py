from __future__ import division
import matplotlib.pyplot as plt


def f(x, w, b):
    '''
    x = input vector
    w = weight vector
    b = bias
    '''
    return b + sum(map(lambda (x, y): x*y, zip(x[:len(x)-1], w)))


def l(w, b, data):
    return (1/len(data)) * \
        sum(map(lambda d: (f(d, w, b) - d[-1])**2, data))


def l_db(w, b, data):
    return (2/len(data)) * \
        sum(map(lambda d: f(d, w, b) - d[-1], data))


def l_dw(w, b, data):
    tot = [0 for _ in range(len(data[0])-1)]
    for triple in data:
        diff = f(triple, w, b) - triple[-1]
        for i in range(len(tot)):
            tot[i] += triple[i]*diff
    for i in range(len(tot)):
        tot[i] = tot[i] * (2/len(data))
    return tot


def parse_data(filename):
    ret = []
    with open(filename) as f:
        for line in f.readlines():
            ret.append(map(float, line.split(',')))
    return ret

training_data = parse_data('data/data-train.csv')
test_data = parse_data('data/data-test.csv')
w = [42 for _ in range(len(training_data[0])-1)]
b = 0
learning_rate = 0.5
threshold = 0.0001
max_iter = 1000
i = 0
delta_w = [1, 1]
delta_b = 1
loss = []


while abs(delta_w[0]) > threshold and abs(delta_w[1]) > threshold and abs(delta_b) > threshold and not i > max_iter:
    # Update w and b
    delta_w = map(lambda x: learning_rate*x, l_dw(w, b, training_data))
    w = map(lambda (x, delta): x - delta, zip(w, delta_w))
    delta_b = learning_rate * l_db(w, b, training_data)
    b -= delta_b
    print i, w, b, l(w, b, training_data)
    loss.append(l(w, b, training_data))
    i += 1

plt.title('Loss change during training')
plt.xlabel('Iteration')
plt.ylabel('Loss value')
plt.axvline(5, color='r', linestyle='--')
plt.axvline(10, color='r', linestyle='--')
plt.plot(loss)
plt.show()
