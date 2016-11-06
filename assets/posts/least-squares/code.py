import numpy as np
import matplotlib.pyplot as plt

def train_coefficients(X, Y):
    return np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(Y)

def predict(X, beta):
    return X.dot(beta)

def sample_line(intercept, alpha, sample_size=20):
    x = np.random.uniform(low=-5, high=5, size=sample_size)
    y = intercept + alpha * x

    x_noise = np.random.randn(sample_size)
    y_noise = np.random.randn(sample_size)

    return x + x_noise, y + y_noise

def rss(beta, y, x):
    return (y - x.dot(beta)).T.dot(y - x.dot(beta))

# Line Plot

x, y = sample_line(3.952, .7432, sample_size = 100)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, y, label="line sample")

plt.xlabel("x")
plt.ylabel("y")

plt.legend(loc="lower right")

plt.grid()


# Train coefficients:

beta = train_coefficients(np.column_stack((ones(100), x)), y)

new_x = np.linspace(-5, 5, 100)
y_hat = predict(np.column_stack((np.ones(100), new_x)), beta)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, y, label="line sample")

plt.xlabel("x")
plt.ylabel("y")

ax.plot(new_x, y_hat, label="fitted line")

plt.legend(loc="lower right")

plt.grid()


# Parabola
SAMPLE_SIZE = 100

parameters = np.asarray([11.12, np.e, 1])

def parabola(parameters, x):
    c = parameters[0]
    b = parameters[1]
    a = parameters[2]

    return a * np.power(x, 2) + b * x + c

fig = plt.figure()
ax = fig.add_subplot(111)

x = np.linspace(-5, 10, SAMPLE_SIZE)
y = parabola(parameters, x)

samples = np.random.uniform(-5, 10, SAMPLE_SIZE)
sigma = 8
y_sampled = parabola(parameters, samples) + sigma * np.random.randn(SAMPLE_SIZE)

ax.plot(x, y, label='true parabola')
ax.scatter(samples, y_sampled, label='samples')
plt.legend(loc="upper left")

plt.grid()

# Preparing data

x_train = np.column_stack((np.ones(SAMPLE_SIZE), x, np.power(x, 2)))
beta = train_coefficients(x_train, y)

print beta

# Ellipse example
(xc, yc) = (6, 12)

t = np.linspace(-np.pi, np.pi, 500)

x = xc +a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi)
y = yc + a*np.cos(t)*np.sin(phi) + b *np.sin(t)*np.cos(phi)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label="Ellipse")
ax.plot(xc, yc, 'or', label="Center of Ellipse")
plt.legend(loc="upper left")
plt.grid()
