import numpy as np
from math import sin
from random import uniform

data = (
    0.0,6.4092654674,
    1.0,7.020086954,
    2.0,5.5853591262,
    3.0,8.3410701973,
    4.0,11.7963142689,
    5.0,12.5416607414,
    6.0,6.2333163282,
    7.0,8.8456094694,
    8.0,12.1798006785,
    9.0,14.6479983038,
    10.0,10.1864773245,
    11.0,15.1032335974,
    12.0,16.6965659555,
    13.0,17.2242214324,
    14.0,16.9795771122,
    15.0,21.3780551384,
    16.0,17.6624273242,
    17.0,22.795301095,
    18.0,23.9824147573,
    19.0,20.8545829905,
    20.0,27.4532893721,
    21.0,28.9149009745,
    22.0,29.5891599774,
    23.0,28.5117587858,
    24.0,25.7408558174,
    25.0,31.6433799072,
    26.0,30.2336468112,
    27.0,32.6406494944,
    28.0,30.405484583,
    29.0,30.728391576,
    30.0,34.7039744911,
    31.0,38.7057368539,
    32.0,34.2692504726,
    33.0,36.5452772562,
    34.0,37.716190198,
    35.0,36.2861174458,
    36.0,43.4079335373,
    37.0,43.7846638025,
    38.0,39.8796452107,
    39.0,41.7241638682,
    40.0,41.9492433956,
    41.0,48.2134435209,
    42.0,49.0644574946,
    43.0,48.7048999898,
    44.0,46.8783057661,
    45.0,48.9507777592,
    46.0,50.2807093747,
    47.0,50.5263656022,
    48.0,55.8248731291,
    49.0,56.8388832768)


def poly_func(x):
    return sin(float(x) ** 0.5) * 0.4 + 0.6 * float(x)


def generate_poly_dots(start_x, end_x, steps, variance, func):
    step = (float(end_x) - float(start_x)) / float(steps)
    x = start_x;
    vec_x = []
    vec_y = []
    for i in range(steps):
        vec_x.append(x)
        r = uniform(-1.0, 1.0) * variance
        vec_y.append(func(x) + r)
        x += step;
    return (vec_x, vec_y)


def gradient_descent2(x, y, theta, alpha, num_iters):
    m = y.size
    for i in range(num_iters):
        hypothesis = x.dot(theta).flatten()
        loss = hypothesis - y
        errors_w = [
            loss * x[:, 0],
            loss * x[:, 1]]
        gradient = [
            errors_w[0].sum() / float(m),
            errors_w[1].sum() / float(m)]
        theta[0][0] = theta[0][0] - alpha * gradient[0]
        theta[1][0] = theta[1][0] - alpha * gradient[1]
    return theta


def gradient_descent(x, y, theta, alpha, num_iters):
    m = y.size
    xt = x.transpose()
    coef = alpha * (1.0 / float(m))
    for i in range(num_iters):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        tmp = np.dot(xt, loss) * coef
        print 'tmp=', tmp
        print 'theta=', theta
        theta = theta - tmp
        print 'res theta=', theta

    return theta


x = np.ones(shape=(len(data) / 2, 2))
x[:, 1] = np.array(data[::2])

# theta = np.zeros(shape=(2, 1))
# y = np.array(data[1::2])
# print gradient_descent2(x, y, theta, 0.0005, 1)

theta = np.zeros(2).reshape((2, 1))
y = np.array(data[1::2]).reshape((len(data) / 2), 1)
res = gradient_descent(x, y, theta, 0.0005, 1)
# print res
