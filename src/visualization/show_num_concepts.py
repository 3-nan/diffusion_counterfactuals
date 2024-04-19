import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Apply the default theme
sns.set_theme()

num_concepts = [1, 10, 20, 50, 100, 200, 300]

fids = [43.41, 43.24, 43.82, 44.8, 45.37, 45.55, 45.68]

l1s = []
l2s = []

flip_rats = [0.286, 0.628, 0.748, 0.854, 0.901, 0.93, 0.935]
confidences = [0.67, 0.744, 0.79, 0.85, 0.88, 0.9, 0.91]

# ticks = np.arange(len(num_concepts))
results_dir = "/results/counterfactuals/imgs"

fig, ax1 = plt.subplots(figsize=(7,3))

color = 'tab:orange'
ax1.set_xlabel('Num. of Concepts')
ax1.set_ylabel('FID', color=color)
ax1.plot(num_concepts, fids, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Flip Ratio', color=color)  # we already handled the x-label with ax1
ax2.plot(num_concepts, flip_rats, color=color)
ax2.tick_params(axis='y', labelcolor=color)

color = 'tab:grey'
ax2.plot(num_concepts, confidences, color=color)

# ax1.grid(False)

# Turns off grid on the secondary (right) Axis.
ax2.grid(False)


# ax = fig.add_subplot(1, 1, 1)
# ax.plot(num_concepts, fids, color='tab:blue')
# ax.plot(num_concepts, flip_rats, color='tab:orange')

# ax.set_xlabel("Num. of Concepts")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "num_concepts.png"))
plt.close()
