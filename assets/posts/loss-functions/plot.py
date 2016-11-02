#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np

sigma = 1

# Original Attributes
GOOD_BEER_ALCOHOL = 7
GOOD_BEER_SWEETNESS = 4

BAD_BEER_1_ALCOHOL = 8
BAD_BEER_1_SWEETNESS = 1

BAD_BEER_2_ALCOHOL = 4
BAD_BEER_2_SWEETNESS = 2

BAD_BEER_3_ALCOHOL = 3
BAD_BEER_3_SWEETNESS = 5

bad_beers = [
    (BAD_BEER_1_ALCOHOL, BAD_BEER_1_SWEETNESS),
    (BAD_BEER_2_ALCOHOL, BAD_BEER_2_SWEETNESS),
    (BAD_BEER_3_ALCOHOL, BAD_BEER_3_SWEETNESS)]

good_alcohol_measurements = sigma * np.random.randn(50) + GOOD_BEER_ALCOHOL
good_sweetness_measurements = sigma * np.random.randn(50) + GOOD_BEER_SWEETNESS

bad_beer_1_alcohol_m = sigma * np.random.randn(50) + BAD_BEER_1_ALCOHOL
bad_beer_1_sweetness_m = sigma * np.random.randn(50) + BAD_BEER_1_SWEETNESS

bad_beer_2_alcohol_m = sigma * np.random.randn(50) + BAD_BEER_2_ALCOHOL
bad_beer_2_sweetness_m = sigma * np.random.randn(50) + BAD_BEER_2_SWEETNESS

bad_beer_3_alcohol_m = sigma * np.random.randn(50) + BAD_BEER_3_ALCOHOL
bad_beer_3_sweetness_m = sigma * np.random.randn(50) + BAD_BEER_3_SWEETNESS

bad_measurements_alcohol = np.concatenate([bad_beer_1_alcohol_m,
                                           bad_beer_2_alcohol_m,
                                           bad_beer_3_alcohol_m])

bad_measurements_sweetness = np.concatenate([bad_beer_1_sweetness_m,
                                             bad_beer_2_sweetness_m,
                                             bad_beer_3_sweetness_m])

good_measurement_points = np.dstack([good_alcohol_measurements, good_sweetness_measurements])[0]
bad_measurement_points = np.dstack([bad_measurements_alcohol, bad_measurements_sweetness])[0]

# First Plot

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(good_alcohol_measurements,
           good_sweetness_measurements,
           marker='o',
           color='orange',
           label="good beer")

ax.scatter(bad_measurements_alcohol,
           bad_measurements_sweetness,
           marker='x',
           color='blue',
           label="other beers")

plt.xlim(0, 10)
plt.ylim(0, 5)

plt.xlabel("alcohol intensity")
plt.ylabel("sweetness")

plt.legend(loc='lower left')

plt.grid()

# Second Plot

def euclidian_distance(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)

def exponential_weight(x, i):
    return np.exp(-(euclidian_distance(x, i) / 2.0))

def estimate_class(point):
    good = sum([exponential_weight(point, i) for i in good_measurement_points]) / len(good_measurement_points)
    bad = sum([exponential_weight(point, i) for i in bad_measurement_points]) / len(bad_measurement_points)

    if (good >= bad):
        return 'GOOD'
    return 'BAD'


def good_beer_sample(n):
    x = sigma * np.random.randn(n) + GOOD_BEER_ALCOHOL
    y = sigma * np.random.randn(n) + GOOD_BEER_SWEETNESS
    return np.dstack([x, y])[0]

def bad_beer_sample(n):

    beers = [np.random.choice(range(len(bad_beers))) for i in range(n)]

    x = [sigma * np.random.randn() +  bad_beers[beers[i]][0] for i in range(n)]
    y = [sigma * np.random.randn() +  bad_beers[beers[i]][1] for i in range(n)]
    return np.dstack([x, y])[0]

sample_size = 50

good_beer_samples = good_beer_sample(sample_size)
bad_beer_samples = bad_beer_sample(sample_size)

true_positive = []
false_negative = []

for i in good_beer_samples:
    if (estimate_class(i) == 'GOOD'):
        true_positive.append(i)
    else:
        false_negative.append(i)

true_negative = []
false_positive = []

for i in bad_beer_samples:
    if (estimate_class(i) == 'BAD'):
        true_negative.append(i)
    else:
        false_positive.append(i)

fig2 = plt.figure()

ax2 = fig2.add_subplot(111)

true_positive = np.asarray(true_positive)
true_negative = np.asarray(true_negative)
false_negative = np.asarray(false_negative)
false_positive = np.asarray(false_positive)

ax2.scatter(true_positive[:,0],
           true_positive[:,1], marker='o',
           color='orange', label='Good beer - OK')

ax2.scatter(true_negative[:, 0],
           true_negative[:, 1], marker='x',
           color='blue', label='Bad beer - OK')

ax2.scatter(false_negative[:, 0],
           false_negative[:, 1], marker='o',
           color='red', label='Good beer - NOK')

ax2.scatter(false_positive[:, 0],
           false_positive[:, 1], marker='x',
           color='red', label='Bad beer - NOK')

plt.xlim(0, 10)
plt.ylim(0, 5)

plt.xlabel("alcohol intensity")
plt.ylabel("sweetness")

plt.legend(loc='lower left')

accuracy = 1.0 * (len(true_negative) + len(true_positive)) / (2 * sample_size)
precision = 1.0 * len(true_positive) / (len(true_positive) + len(false_positive))
recall = 1.0 * len(true_positive) / (len(true_positive) + len(false_negative))
