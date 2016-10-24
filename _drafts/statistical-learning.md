---
layout: post
title: A little bit on Loss Functions
category: Statistical Learning
date: 2016-10-22 19:41:00
author: Flavio Truzzi
---

Since the beginning of college, and my first step on my old lab I was always fascinated
with Machine Learning and Artificial Intelligence in general, and that was the reason I made my master on these topics later on.  

So, no matter what kind of prediction I want to make, which function I want to fit, it will
always end up on minimizing some cost function. Since I am reading an awesome book [TESL](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576 "The Elements of Statistical Learning"){:target="_blank"}
that talks on this subject I thought I could write about it a little bit and keep this as my notes.

Suppose I am seeking a function $$f(X)$$ for predicting $$Y$$. How I measure if I am being
successful on this task?

# Loss Functions

A loss function is a function that measure the empirical error over my training set.
Imagine that I want to be able to classify my favorite beer (Trappistes Rochefort)
only by its flavor and impress my friends with such a refined taste.

My friends defined the following process in order to see me drunk:

  1. I take a sip and make a prediction $$ f(X) $$
  2. If my prediction is correct, I can enjoy my beer or I can pass to the next sip
  3. If my prediction is wrong I need to drink the whole bottle and go to the next sip.

So how can I prepare?

## Definition

I need to define a function $$ L(f(X), Y) $$  that will penalize mistakes in the predictions, there are
infinite ways to define such a function, however we are going to look at some, but first lets take a look on the expectation of
the loss function, which is in fact the function that we want to minimize:

$$ \mathbb{E}(L(x,y)) = \int \int L(x,y) g(x,y) dx\,dy $$

where, $$g(x,y)$$ is the joint probability density function of $$X$$ and $$Y$$.
In the book they use a different notation, that I found very unusual:

$$ \mathbb{E}(L(x,y)) = \int L(x, y) Pr(dx, dy) $$

I found this [great](http://stats.stackexchange.com/questions/67038/confused-by-derivation-of-regression-function) explanation of how to produce the final equation, and I'll transcribe it here:

$$\begin{equation}\begin{split}\mathbb{E}(L(x,y))&=\int\int L(x,y)g(x,y)dx\,dy \\
&=\int\bigg[\int L(x,y)g(y|x)g(x)dy\bigg]dx \\
&=\int\bigg[\int L(x,y)g(y|x)dy\bigg]g(x)dx \\
&=\int\bigg[\mathbb{E}_{Y|X} (L(x,y)\bigg]g(x)dx \\
&=\mathbb{E}_X(\mathbb{E}_{Y|X} (L(x,y)))\end{split}\end{equation}$$



#### $$L_2$$ Loss

$$ L_2(f(X), Y) = \mathbb{E}[(Y - f(X))^2] $$

The solution of the expected value of the loss function is:

$$ f(x) = \mathbb{E}(Y|X=x) $$

that is the conditional expectation. This means that the best prediction of Y at any
point $$X=x$$ is the conditional mean (L2). This solution comes from the theorem that states:

$$ \mathbb{E}[\mathbb{E}[Y|X]] = \mathbb{E} [Y]$$

#### $$L_1$$ Loss

$$ L_1(f(X), Y) = \mathbb{E} |Y - f(X)|$$

In the case of the $$L_1$$ loss the solution is the conditional median:

$$ f(x) = median(Y|X=x) $$

The conditional median is a different measure, and it is more robust to noise, compared
to the conditional average (remember that only one spurious element can make the average move a lot).

However the L1 loss is less used because of the discontinuities in the derivatives.

### 0-1 Loss

The 0-1 Loss is used on categorical classifications, and it pays 1 when the classification is wrong
and 0 when the classification is correct. In this case we can define the loss function being:
$$ L(\hat{G}(X), G) $$, where $$\hat{G}(X)$$ is a function that given the features $$X$$ map it to the an
element in the space $$\mathcal{G}$$, the set of possible classes. In this case, the expectation of the loss functions is:

$$ \mathbb{E}(L(\hat{G}(X), G)) = \mathbb{E}_X \sum_{k=1}^{|\mathcal{G}|} L(\hat{G}(X), G_k) Pr(G_k | X)$$
