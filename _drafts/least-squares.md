---
layout: post
title: Least Squares
category: Statistical-Learning
date: 2016-11-05 19:14:00
author: Flavio Truzzi
cover: https://images.unsplash.com/photo-1462663608395-404cb6246eaf?fit=crop&fm=jpg&h=400&q=100&w=1450
use_math: true
---

Usually, I say that new is always better! However, with so many people talking about deep learning, they forget that some tasks do not require a complex model, and we can still do great with more traditional models. Today I want to talk about how to fit a linear model! If you want you can jump to the [code]().

## Linear Model

A linear model have the following form:

$$
\begin{equation}
f(X) = \beta_0 + \sum_{j = 1}^{P} X_j \beta_j
\end{equation}
$$

Normally when we think of linear model we think on first degree polynomials, but that is not what a linear model means. **A linear model is linear in the inputs** $X^T = (X_1, X_2, \dots, X_p)$, but this does not define how these inputs are defined. They can be:

  - quantitative inputs;
  - transformation of the inputs (e.g., log, sqrt);
  - basis expansions $X_2 = X_1^2$, $X_3 = X_2^3$;
  - numeric representation of qualitative inputs;
  - interactions between variables $X_3 = X_1 X_2$.

## Least Squares

The first method devised to fit this is called Least Squares (or ordinary least squares). It was first published by [Legendre](https://en.wikipedia.org/wiki/Adrien-Marie_Legendre) who published the method in 1805, but it has a little controversy on who is actually the author.

The method was also published in 1808 by an American called [Robert Adrain](https://en.wikipedia.org/wiki/Robert_Adrain), but the most controversy comes from the great [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss) who in 1809 said that he was already using the method since 1795. Usually Gauss is considered the father because he went further by connecting the method with the probability and to the Gaussian Distribution, the method was used to predict the future location of the minor planet Ceres which was discovered on 1st January of 1801. If you are into this kind of controversy [this](http://projecteuclid.org/download/pdf_1/euclid.aos/1176345451) can be a good reading material.

Least Squares work by minimizing the sum of the [L2 Cost Function](/statistical-learning/2016/10/22/a-little-bit-on-loss-functions.html#l2-loss), this is usually called Residual Squared Sum (RSS):

$$
\begin{equation}
\begin{split}
RSS(\beta) &= \sum_{i = 1}^{N} (Y_i - f(X_i))^2 \\
&= \sum_{i = 1}^{N} (y_i - \beta_0 - \sum_{j=1}^{P}X_{ij}\beta_j)^2
\end{split}
\end{equation}
$$

This criterion tell us how much the model does not fit the data. Let's find it's minimum. First, rewriting to its matrix form:

$$
\begin{equation}
RSS(\beta) = (Y - X\beta)^T(Y - X\beta)
\end{equation}
$$

here we are adding a 1 to as first column of $X$, this term is usually called bias. Taking its derivative with respect to $\beta$:

$$
\frac{\partial RSS}{\partial \beta} = -2X^T(Y-X\beta)
$$

Forcing it to be equal zero, and assuming that $X^TX$ is positive definite:

$$
\begin{split}
\frac{\partial RSS}{\partial \beta} &= 0 \\
-2X^T(Y-X\beta) &= 0 \\
X^TX\beta &= X^TY \\
(X^TX)^{-1}X^TX\beta &= (X^TX)^{-1}X^TY \\
\beta &= (X^TX)^{-1}X^TY
\end{split}
$$

With that $\beta$ in hands we can now make predictions for new values of $X$, with:

$$
\begin{equation}
\hat{Y} = X\beta = \underbrace{X(X^TX)^{-1}X^T}_{H}Y
\label{eq:yhat}
\end{equation}
$$

The part $(X^TX)^{-1}X^T$ is sometimes called $H$, or the hat matrix. The reason for this is worthy of a **Ba dum tss**, since it is because it puts the hat on $Y$.

There are some caveats that we need to keep in mind when using this. First, $X$ may not be full ranked, i.e., maybe there are linear dependents inputs, if that is the case it is important to remove those, it is quite common when working with images since the number of parameters $p$ may be much bigger than the number of test cases for the training. Note that this can also be tackled with regularization and other methods, which are not the scope of this post.

## Code

Here we are going to use python and numpy to make the predictions, obviously there are other packages that you can use, or even implement yourself. Scikit learn also have implementations for it, but in this case I just want to code it using only numpy.

{% highlight python %}
import numpy as np

def train_coefficients(X, Y):
    return np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(Y)

def predict(X, beta):
    return X.dot(beta)
{% endhighlight %}

I am using the `pinv` function instead of the `inv` in order to tackle singular matrices.

Now, lets create some examples and check how well this actually behaves.

### Fitting a Line

Let's generate some points that follow a line and add a gaussian error on it.

{% highlight python %}
def sample_line(intercept, alpha, sample_size=20):
    x = np.random.uniform(low=-5, high=5, size=sample_size)
    y = intercept + alpha * x

    x_noise = np.random.randn(sample_size)
    y_noise = np.random.randn(sample_size)

    return x + x_noise, y + y_noise

# Line Plot

x, y = sample_line(3.952, .7432, sample_size = 100)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(x, y, label="line sample")

plt.xlabel("x")
plt.ylabel("y")

plt.legend(loc="lower right")

plt.grid()
{% endhighlight %}

I choose the value `3.952` as intercept value and `0.7432` as coefficient. This generated the following plot:

![Sample Line](/assets/posts/least-squares/line_sample.png)

Finally, we can train the coefficients with the code defined above, and plot it:

{% highlight python %}

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

{% endhighlight %}

Please note that we are adding the bias on x before calling the function `train_coefficients` and also before calling the `predict`. The result of this is as follows:

![Fitted Line](/assets/posts/least-squares/fitted_line.png)

The fitted parameters are: $\hat{\beta}=\begin{bmatrix} 3.87515291 &  0.71652639 \end{bmatrix}^T$. You may think that this is not quite similar to the original $\beta=\begin{bmatrix}3.952 & 0.7432\end{bmatrix}$, but considering that we added noise on $X$ and $Y$, it looks actually pretty good.


### Fitting a Parabola

Now, let's try this with a parabola. First, we need to generate a parabola and generate some samples. I choose the following parabola:

$$
y = 12 + ex + x^2
$$

And its plot with the sampled data:

![Parabola](/assets/posts/least-squares/parabola.png)

The plot was generated with the following code:

{% highlight python %}
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
{% endhighlight %}

In order to fit this, we need to "fix" our X (like we did in the previous example by adding the bias), in this case we need to add the bias, and its square, so every point in our training data looks like: $X_i = \begin{bmatrix} 1 & x & x^2 \end{bmatrix} $.

{% highlight python %}
x_train = np.column_stack((np.ones(SAMPLE_SIZE), x, np.power(x, 2)))
beta = train_coefficients(x_train, y)
{% endhighlight %}

After the training we got: $\beta = \begin{bmatrix} 11.12 & 2.71828183 & 1.\end{bmatrix}$, a perfect match :)

### Fitting an ellipse

As usual, first the definition of an ellipse, in its parametric form:

$$
\begin{equation}
x(t) = x_c + a \, cos(t) \, cos(\phi) - b \, sin(t) \, sin(\phi)
\label{eq:ellipsex}
\end{equation}
$$

$$
\begin{equation}
y(t) = y_c + a \, cos(t) \, sin(\phi) - b \, sin(t) \, cos(\phi)
\label{eq:ellipsey}
\end{equation}
$$

where $(x_c, y_c)$ is the center of the ellipse and $\phi$ is the angle between the $X$-axis and the major axis of the ellipse.

Let's create an ellipse, I defined that the center will be $\begin{bmatrix}xc& yc\end{bmatrix} = \begin{bmatrix}6& 12\end{bmatrix} $ and $\phi = -\frac{\pi}{8}$.

{% highlight python %}
# Ellipse example
(xc, yc) = (6, 12)

t = np.linspace(-np.pi, np.pi, 500)
phi = -np.pi / 8

x = xc +a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi)
y = yc + a*np.cos(t)*np.sin(phi) + b *np.sin(t)*np.cos(phi)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x, y, label="Ellipse")
ax.plot(xc, yc, 'or', label="Center of Ellipse")
plt.legend(loc="upper left")
plt.grid()
{% endhighlight %}

The result is the following plot:

![Ellipse](/assets/posts/least-squares/ellipse.png)

One thing that is important here is that if we look at equations $\eqref{eq:ellipsex}$ and $\eqref{eq:ellipsey}$ is that the parameters are not linear ($a\,cos(\phi)$, $b\,sin(\phi)$, $a\,sin(\phi)$ and $b\,cos(\phi)$).

Hence we are going to fit the parameters in a different way, instead of fitting:

$$
\beta = \begin{bmatrix} X_c & Y_c & a & b & \phi \end{bmatrix}
$$

we are going to fit the following parameters:

$$
\beta = \begin{bmatrix} X_c & Y_c & a\,cos(\phi) & a\,sin(\phi) & b\,cos(\phi) & b\,sin(\phi) \end{bmatrix}
$$

and afterwards, calculate $a$, $b$ and $phi$.
