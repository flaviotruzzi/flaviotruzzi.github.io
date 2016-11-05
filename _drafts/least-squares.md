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

Normally when we think of linear model we think on first degree polynomials, but that is not what a linear model means. A linear model is linear in the inputs $X^T = (X_1, X_2, \dots, X_p)$, but this does not define how these inputs are defined. They can be:

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
\hat{Y} = X\beta = \underbrace{X(X^TX)^{-1}X^T}_{H}Y
$$

The part $(X^TX)^{-1}X^T$ is sometimes called $H$, or the hat matrix. The reason for this is worthy of a **Ba dum tss*, since it is because it puts the hat on $Y$.

There are some caveats that we need to keep in mind when using this. First, X may not be full ranked, i.e., maybe there are linear dependents inputs, if that is the case it is important to remove those, it is quite common when working with images since the number of parameters $p$ may be much bigger than the number of test cases for the training. Note that this can also be tackled with regularization and other methods, which are not the scope of this post.

## Code
