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

fids_37 = [50.38, 44.16, 44.78, 45.45, 45.84]
flip_rats_37 = [0.311, 0.721, 0.837, 0.894, 0.935]

fids_resnet = [43.96, 44.4, 45.29, 45.87, 46.6]
flip_rats_resnet = [0.66, 0.762, 0.858, 0.902, 0.936]


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

plt.figure()

plt.scatter(flip_rats, fids)
plt.scatter(flip_rats_37, fids_37, color='orange')
plt.scatter(flip_rats_resnet, fids_resnet, color='green')

for (xi, yi, nc) in zip(flip_rats, fids, num_concepts):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=8)

for (xi, yi, nc) in zip(flip_rats_37, fids_37, num_concepts[1:6]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=8)

for (xi, yi, nc) in zip(flip_rats_resnet, fids_resnet, num_concepts[1:6]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=8)

plt.xlabel('Flip Ratio')
plt.ylabel('FID')
plt.gca().invert_yaxis()
plt.legend(['VGG16bn_40', 'VGG16bn_37', 'ResNet18'])
plt.savefig(os.path.join(results_dir, "relative_fid_flips.png"))
plt.close()
