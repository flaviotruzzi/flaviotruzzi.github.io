---
layout: post
title: A little bit on Loss Functions
category: Statistical-Learning
date: 2016-10-22 19:41:00
author: Flavio Truzzi
cover: https://images.unsplash.com/photo-1436076863939-06870fe779c2?fit=crop&fm=jpg&h=400&q=100&w=1450
use_math: true
---

Since the beginning of college, and my first step on my old lab, I was always fascinated with Machine Learning and Artificial Intelligence in general, and that was the reason I made my master on these topics later on.  

So, no matter what kind of prediction I want to make, which function I want to fit, it will always end up on minimizing some cost function. Since I am reading an awesome  [book](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576 "The Elements of Statistical Learning"){:target="_blank"} that talks about this subject I thought I could write about it a little bit and keep this as my notes.

Suppose I am seeking a function $f(X)$ for predicting $Y$. How can we measure success on this task?

# Loss Functions

A loss function is a function that measures the empirical error over a training set. Imagine that I want to classify my favorite beer *(Trappistes Rochefort 10)* only by its flavor.

In order to make things more interesting my friends defined the following process:

  1. I take a sip and make a prediction $ f(X) $;
  2. If my prediction is correct, I can enjoy my beer or I can pass to the next sip;
  3. If my prediction is wrong I need to drink the whole bottle and go to the next sip.

This is a classification task. The aim is to be able to classify if I am sipping or not my favorite beer 100 out 100 times.

In this setting, when I misclassify I am forced to drink the whole bottle of beer.  And, despite some fellow german colleagues may think, that is not the goal here.

Every time I make a wrong prediction I am forced to drink the  wrong beer. Getting drunk with the bad beer is a negative reinforcement whilst drinking the good beer is a positive reinforcement.

## Definition

I need to define a function $ L(f(X), Y) $  that will penalize mistakes in the predictions. There are infinite ways to define such a function, but we are going to look only at some of them.

First let's take a look on the expectation of the loss function, which is, in fact, the function that we want to minimize. In the book it is defined as:

$$
\begin{equation}
  \mathbb{E}(L(f(X),Y)) = \int L(f(x), y) Pr(dx, dy)
\end{equation}
$$

$$Pr(dX, dY)$$ is the joint probability density function of $$X$$ and $$Y$$. We can rewrite this as:

$$
\begin{equation}
  \mathbb{E}(L(f(X),Y)) = \int \int L(f(x),y) p(x,y) dx\,dy
\end{equation}
$$

where, $$p(x,y)$$ is the joint probability density function of $$X$$ and $$Y$$.

I found [this explanation](http://stats.stackexchange.com/questions/67038/confused-by-derivation-of-regression-function) of the meaning of the $$Pr(dx, dy)$$ notation and how to derive the final equation. I'll just transcribe it here:

$$
\begin{equation}
  \begin{split}
    \mathbb{E}(L(f(x),y))&=\int\int L(f(x),y)p(x,y)dx\,dy \\
    &=\int\bigg[\int L(f(x),y)p(y|x)p(x)dy\bigg]dx \\
    &=\int\bigg[\int L(f(x),y)p(y|x)dy\bigg]p(x)dx \\
    &=\int\bigg[\mathbb{E}_{Y|X} (L(f(x),y)\bigg]p(x)dx \\
    &=\mathbb{E}_X(\mathbb{E}_{Y|X} (L(f(x),y)))
  \end{split}
  \label{eq:expectation}
\end{equation}
$$

#### $L_2$ Loss

The $L_2$ loss is defined as the quadratic error between the prediction $f(X)$ and the correct value $Y$:

$$ L_2(f(X), Y) = (Y - f(X))^2 $$

If we apply $ L_2 $ into equation $\eqref{eq:expectation}$ we get:

$$
\mathbb{E}(L(x,y)) = \mathbb{E}_X(\mathbb{E}_{Y|X} ( (Y - f(X))^2 ))
$$

We know that the function above is minimal when $Y = f(x)$, i.e. if we minimize the function pointwise we get the minimal value for the expected error. We can write $f(x)$ as:

$$
f(x) = \text{argmin}_c \mathbb{E}_{Y|X} ([Y-c]^2 | X = x)
$$

The solution of the expected value of the loss function is:

$$ f(x) = \mathbb{E}(Y|X=x) $$

that is the conditional expectation. This means that the best prediction of Y at any
point $$X=x$$ is the conditional mean (L2).

{::comment}
This solution comes from the theorem that states:

$$ \mathbb{E}[\mathbb{E}[Y|X]] = \mathbb{E} [Y]$$)
{:/comment}

#### $L_1$ Loss

The $L_1$ is defined as the absolute error between the prediction $f(X)$ and the correct value $Y$:

$$ L_1(f(X), Y) = \mathbb{E} |Y - f(X)|$$

If we apply $ L_1 $ into equation $\eqref{eq:expectation}$ we get:

$$
\begin{equation}
  \mathbb{E}(L(x,y)) = \mathbb{E}_X(\mathbb{E}_{Y|X} ((|Y - f(x)|)| X))
\end{equation}
\label{eq:l1}
$$

When $f(X_i) \neq Y_i$ the derivative of $L_1$ is:

$$
\begin{equation}
  \begin{split}
    \frac{d}{dx} |Y_i - f(X_i)| &= - \text{sign}(Y_i - f(X_i)) \\
    &= \begin{cases}
        -1,              & \text{if } f(X_i) < Y_i\\
        +1,              & \text{if } f(X_i) > Y_i
       \end{cases}
  \end{split}
\end{equation}
$$

If we take the derivative of the expectation $\eqref{eq:l1}$ and equal it to zero, we get that our error decreases if we choose $f(X)$ to even out the $+1$ and the $-1$. In this case the solution is the conditional median:

$$ f(x) = median(Y|X=x) $$

The conditional median is a different measure. It is more robust to noise compared to the conditional average (remember that only one spurious element can make the average move ). Still, the L1 loss is less used because of the discontinuities in the derivatives.

### 0-1 Loss

The 0-1 Loss is used on categorical classifications. It yields a loss of 1 when the classification is wrong and 0 when the classification is correct.

In this case we can define the loss function as: $ L(\hat{G}(X), G) $. The function $$\hat{G}(X)$$ maps a given set of features $X$ to an element in the space $\mathcal{G}$, i.e., the set of possible classes. In this case, the expectation of the loss functions is:

$$ \mathbb{E}(L(\hat{G}(X), G)) = \mathbb{E}_X \sum_{k=1}^{|\mathcal{G}|} L(\hat{G}(X), G_k) Pr(G_k | X)$$

If we use the 0-1 loss there:

$$ L(\hat{G}(X), G) = \begin{cases}
    1,              & \text{if } \hat{G}(X) \neq G\\
    0,              & \text{otherwise}
\end{cases} $$

we get as solution:

$$
\begin{equation}
  \begin{split} \hat{G}(X) &= \mathcal{G}_k \text{ if } Pr(\mathcal{G}|X=x) \\
 &= \max_{g \in \mathcal{G}} Pr(g|X=x) \end{split}
\end{equation}
\label{eq:01loss}
 $$


This is known as the Bayes classifier, which classify to the most probable class.

### Back to the beer

Let's get back to our initial motivation. How can we be able to discover which beer I am sipping without getting drunk?

Suppose that my palate is only able to identify two different features: alcohol intensity and sweetness. Consider that alcohol intensity scales from 0 to 10, and sweetness from 0 to 5.

Since my palate is not 100% accurate I have some errors on my measurements, caused either by taste buds saturation or by the alcohol interference. I know that the alcohol intensity of my favorite beer is 7 and the sweetness is 4. After some sips I plotted my measurements:

![Measurements](/assets/posts/loss-functions/measurements.png)

Here we have 4 different beers with different characteristics. The code to generate this is available [here]().

How can I use the result of equation $\eqref{eq:01loss}$ to find the beer that I am sipping? If we don't know anything about the underlying distributions, we need to figure out some way to approximate the probability $Pr$.

Hence, I'll imagine at first that my measurements follow a gaussian distribution. The probability density function is as follows:

$$
\begin{equation}
pdf(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\sigma^2\pi}}\exp\bigg({\frac{-(x - \mu)^2}{2\sigma^2}}\bigg)
\end{equation}
$$

where $\mu$ is the expectation of the distribution and $\sigma$ is the standard deviation. This equation tells that the probability decreases exponentially with the quadratic of difference between the point $x$ and $\mu$.

With that in mind we can create an approximation for the probability:

$$
 \widehat{Pr}(g | X=x) = \frac{1}{|g|} \sum_{i \in g} \exp\bigg( -\frac{d(X,i)}/2 \bigg)
$$

where $d$ is the euclidian distance between point $X=x$ and point $i$. With this approximation we can plug $\widehat{Pr}$ back in equation $\eqref{eq:01loss}$.

Note that in the equation above we give much more importance to points that are near. You can imagine this as a k-nearest neighbors (which is, in fact, another approximation to $\eqref{eq:01loss}$).


So in order to test this classification procedure, I generated 50 points with the good beer distribution, and 50 points with one of the bad beers distribution (the distribution was randomly chosen). After that I applied our classification procedure that we defined above. Here are some stats:

* Accuracy: 91%
* Precision: 87.27%
* Recall: 96%

Here is a plot of this evaluation:

![Evaluation](/assets/posts/loss-functions/classification.png)

In the plot above the circle orange points are the ones that my predictor was able to correctly classify as the good beer, the blue crosses are the ones that the predictor was able to classify as the bad beer. The red circles are the false negatives, i.e. the good beer that I was unable to predict correctly, and the red crosses are the false positives, i.e, the bad beer that I wrongly assumed to be good.

I hope you liked the long post.

Cheers!
