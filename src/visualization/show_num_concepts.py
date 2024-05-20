import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colormaps as cmaps
import seaborn as sns

# Apply the default theme
sns.set_theme(style="whitegrid", palette="pastel", rc={'xtick.bottom': True})
# tab10 = cmaps['tab10'] #.resampled(10)
tab10 = cmaps["tab20"]

same_size = (6.4, 4.3)
num_concepts = [10, 20, 50, 100, 200, 300]

fids = [43.24, 43.82, 44.8, 45.37, 45.55, 45.68]

l1s = []
l2s = []

# fid_base = 45.14
# flip_rat_base = 0.94
fid_base = 46.2
flip_rat_base = 0.942

flip_rats = [0.628, 0.748, 0.854, 0.901, 0.93, 0.935]
confidences = [0.744, 0.79, 0.85, 0.88, 0.9, 0.91]

fids_40_optim = [43.38, 44.04, 44.9, 45.42, 45.56]
flip_rats_40_optim = [0.797, 0.84, 0.89, 0.915, 0.934]

fids_40_spatial_optim = [42.67, 43.42, 43.84, 44.32, 44.73]
flip_rats_40_spatial_optim = [0.739, 0.788, 0.844, 0.875, 0.895]

fids_37 = [43.67, 44.16, 44.78, 45.45, 45.84]
flip_rats_37 = [0.586, 0.721, 0.837, 0.894, 0.935]

fids_37_optim = [43.94, 44.43, 44.97, 45.54, 45.94]
flip_rats_37_optim = [0.767, 0.821, 0.883, 0.93, 0.946]

fids_37_spatial_optim = [43.29, 43.74, 44.25, 44.62, 45.01]
flip_rats_37_spatial_optim = [0.735, 0.788, 0.843, 0.875, 0.901]

fids_resnet = [43.96, 44.4, 45.29, 45.87, 46.6]
flip_rats_resnet = [0.66, 0.762, 0.858, 0.902, 0.936]

fids_resnet_optim = [44.2, 44.86, 45.58, 46.13, 46.72]
flip_rats_resnet_optim = [0.806, 0.846, 0.901, 0.925, 0.944]

fids_resnet_spatial_optim = [43.8, 43.73, 44.74, 45.32, 45.74]
flip_rats_resnet_spatial_optim = [0.778, 0.826, 0.886, 0.909, 0.927]


# ticks = np.arange(len(num_concepts))
results_dir = "/results/counterfactuals/imgs"

fig, ax1 = plt.subplots(figsize=same_size)  # plt.subplots(figsize=(5.5,3))

color = tab10(0)
ax1.set_xlabel('Num. of Concepts')
ax1.set_ylabel('FID', color=color, fontsize=14)
ax1.axhline(y=fid_base, color=color, linestyle='--')
# ax1.plot(num_concepts, fids, color=color)
# ax1.scatter(num_concepts, fids, color=color)
ax1.plot(num_concepts[:-1], fids_40_optim, color=color)
ax1.scatter(num_concepts[:-1], fids_40_optim, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax1.set_ylim(42.5, 46.5)
plt.locator_params(axis='y', nbins=6)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax1.invert_yaxis()

color = tab10(2)
ax2.set_ylabel('Flip Ratio', color=color, fontsize=14)  # we already handled the x-label with ax1
ax2.axhline(y=flip_rat_base, color=color, linestyle='--')
# ax2.plot(num_concepts, flip_rats, color=color)
# ax2.scatter(num_concepts, flip_rats, color=color)
ax2.plot(num_concepts[:-1], flip_rats_40_optim, color=color)
ax2.scatter(num_concepts[:-1], flip_rats_40_optim, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# color = 'tab:grey'
# ax2.plot(num_concepts[:-1], confidences[:-1], color=color)

# ax1.grid(False)

# Turns off grid on the secondary (right) Axis.
ax2.grid(False)

ax2.set_ylim(0.79, 0.955)
plt.locator_params(axis='y', nbins=6)

line1 = Line2D([0], [0], label='CoLa-DCE', color='gray', linestyle='-')
line2 = Line2D([0], [0], label='Baseline', color='gray', linestyle='--')

plt.legend([line1, line2], ['CoLa-DCE (Ours)', 'Baseline'], loc='center right', fontsize=14)
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(num_concepts, fids, color='tab:blue')
# ax.plot(num_concepts, flip_rats, color='tab:orange')

# ax.set_xlabel("Num. of Concepts")
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "num_concepts.png"))
plt.savefig(os.path.join(results_dir, "num_concepts.svg"))
plt.close()

plt.figure(figsize=same_size)

linewidth = 1.

plt.plot(flip_rats_37_optim[:-1], fids_37_optim[:-1], color=tab10(0), linewidth=linewidth)    # orange
plt.scatter(flip_rats_37_optim[:-1], fids_37_optim[:-1], color=tab10(0))
plt.plot(flip_rats_37_spatial_optim[:-1], fids_37_spatial_optim[:-1], color=tab10(1), linewidth=linewidth)
plt.scatter(flip_rats_37_spatial_optim[:-1], fids_37_spatial_optim[:-1], color=tab10(1))
# plt.plot(flip_rats[:5], fids[:5], linewidth=linewidth)
# plt.scatter(flip_rats[:5], fids[:5])

plt.plot(flip_rats_40_optim[:-1], fids_40_optim[:-1], color=tab10(2), linewidth=linewidth)
plt.scatter(flip_rats_40_optim[:-1], fids_40_optim[:-1], color=tab10(2))
plt.plot(flip_rats_40_spatial_optim[:-1], fids_40_spatial_optim[:-1], color=tab10(3), linewidth=linewidth)       # red
plt.scatter(flip_rats_40_spatial_optim[:-1], fids_40_spatial_optim[:-1], color=tab10(3))

plt.plot(flip_rats_resnet_optim[:-1], fids_resnet_optim[:-1], color=tab10(4), linewidth=linewidth)     # green
plt.scatter(flip_rats_resnet_optim[:-1], fids_resnet_optim[:-1], color=tab10(4))
plt.plot(flip_rats_resnet_spatial_optim[:-1], fids_resnet_spatial_optim[:-1], color=tab10(5), linewidth=linewidth)     # green
plt.scatter(flip_rats_resnet_spatial_optim[:-1], fids_resnet_spatial_optim[:-1], color=tab10(5))

plt.ylim(42.5, 46.5)

h = plt.gca().get_children()

for (xi, yi, nc) in zip(flip_rats_37_optim[:-1], fids_37_optim[:-1], num_concepts[:5]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=10)
for (xi, yi, nc) in zip(flip_rats_37_spatial_optim[:-1], fids_37_spatial_optim[:-1], num_concepts[:5]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=10)

for (xi, yi, nc) in zip(flip_rats_40_optim[:-1], fids_40_optim[:-1], num_concepts[:5]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=10)
for (xi, yi, nc) in zip(flip_rats_40_spatial_optim[:-1], fids_40_spatial_optim[:-1], num_concepts[:5]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=10)

for (xi, yi, nc) in zip(flip_rats_resnet_optim[:-1], fids_resnet_optim[:-1], num_concepts[:5]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=10)
for (xi, yi, nc) in zip(flip_rats_resnet_spatial_optim[:-1], fids_resnet_spatial_optim[:-1], num_concepts[:5]):
    plt.text(xi, yi, nc, va='bottom', ha='center', fontsize=10)

legend1 = plt.legend(h[1::4], [r'$VGG16_{bn}37$', r'$VGG16_{bn}40$', r'$ResNet18$'], fontsize=14)
# plt.legend(h[1::2], [r'$VGG16_{bn}37$', r'$VGG16_{bn}37_{spatial}$', r'$VGG16_{bn}40$', r'$VGG16_{bn}40_{spatial}$', r'$ResNet18$', r'$ResNet18_{spatial}$'], fontsize=14)
# plt.legend(h[1::2], [r'$VGG16_{bn}37$', r'$VGG16_{bn}37_{spatial}$', r'$ResNet18$', r'$ResNet18_{spatial}$'], fontsize=14)

line1 = Line2D([0], [0], label='default', color='gray', linestyle='-')
line2 = Line2D([0], [0], label='spatial', color='lightgray', linestyle='-')
plt.legend([line1, line2], ['default', 'spatial'], loc='center right', fontsize=14)

plt.gca().add_artist(legend1)

plt.locator_params(axis='y', nbins=6)
plt.xlabel('Flip Ratio', fontsize=14)
plt.ylabel('FID', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
# plt.legend(h[1::2], ['VGG16bn_37', 'VGG16bn_40', 'ResNet18', 'optimized'])
plt.savefig(os.path.join(results_dir, "relative_fid_flips.png"))
plt.savefig(os.path.join(results_dir, "relative_fid_flips.svg"))
plt.close()
