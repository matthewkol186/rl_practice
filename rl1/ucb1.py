import numpy as np

arms = np.array([0.1, 0.2, 0.3])
sample_means = np.array([1.0, 1.0, 1.0])
sample_cts = np.array([1.0, 1.0, 1.0])

NUM_TO_PULL = 100000
total_reward = 0

for i in range(NUM_TO_PULL):
    ind = np.argmax(np.array([x + np.sqrt(2 * np.log(np.sum(sample_cts)) / sample_cts[i])  for i, x in enumerate(sample_means)]))
    reward = 1 if np.random.random() < arms[ind] else 0
    total_reward += reward
    sample_cts[ind] += 1
    sample_means[ind] = (1 - 1 / sample_cts[ind]) * sample_means[ind] + (1 / sample_cts[ind]) * reward

print(sample_means) 
print(total_reward)