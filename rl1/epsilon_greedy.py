import numpy as np

arms = np.array([0.1, 0.2, 0.3])
sample_means = np.array([0.0, 0.0, 0.0])
sample_cts = np.array([0.0, 0.0, 0.0])

NUM_TO_PULL = 100000
EPS = 0.05
# can modify EPS to be 1/t
total_reward = 0

for i in range(NUM_TO_PULL):
    ind = np.random.choice(3, 1)[0] if np.random.random() < EPS else np.argmax(sample_means)
    reward = 1 if np.random.random() < arms[ind] else 0
    total_reward += reward
    sample_cts[ind] += 1
    sample_means[ind] = (1 - 1 / sample_cts[ind]) * sample_means[ind] + (1 / sample_cts[ind]) * reward

print(sample_means) 
print(total_reward)