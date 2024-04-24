import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_violin_plot():

    sns.set_theme()

    result_dir = '/results/counterfactuals'

    modelpath = 'resnet18'

    all_data = []
    all_random_data = []

    if modelpath.startswith('resnet'):
        concept_nums = [10, 20, 50, 100, 200]
    else:
        concept_nums = [1, 10, 20, 50, 100, 200, 300]

    for n_concepts in concept_nums:

        if os.path.isfile(f'/results/counterfactuals/validity/validity_{modelpath}_concept_{n_concepts}_concept.npy'):
            quality_ratios = np.load(f'/results/counterfactuals/validity/validity_{modelpath}_concept_{n_concepts}_concept.npy')
            random_ratios = np.load(f'/results/counterfactuals/validity/validity_{modelpath}_concept_{n_concepts}_random.npy')

            all_data.append(quality_ratios)
            all_random_data.append(random_ratios)

    plt.figure(figsize=(6.4, 4))

    qplot = plt.violinplot(all_data, showmeans=True, showextrema=False)
    rplot = plt.violinplot(all_random_data, showmeans=True, showextrema=False)

    qcolor = "blue"

    for pc in qplot['bodies']:
        pc.set_facecolor(qcolor)
        pc.set_edgecolor(qcolor)

    qplot['cmeans'].set_facecolor(qcolor)
    qplot['cmeans'].set_edgecolor(qcolor)

    for pc in rplot['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('gray')

    rplot['cmeans'].set_facecolor('gray')
    rplot['cmeans'].set_edgecolor('gray')

    plt.legend([qplot['bodies'][0], rplot['bodies'][0]], ['CoLa-DCE', 'Random'], loc='lower right')

    plt.xticks(np.arange(1, len(concept_nums)+1), concept_nums)
    plt.xlabel('Num. Concepts')
    plt.ylabel('Attribution Ratio\n(w.r.t highest-attributed concepts)')

    plt.tight_layout()
    plt.savefig(f'/results/counterfactuals/validity/violin_{modelpath}.png')
    plt.close()


if __name__ == '__main__':

    make_violin_plot()
