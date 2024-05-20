""" Comparing 2 classes by clustering.
    Visualizing relation of new point to clusters.
    Visualizing position of counterfactual in clusters.
"""
import os
import h5py
import hydra
from corelay.pipeline.spectral import SpectralClustering
from corelay.processor.affinity import SparseKNN
from corelay.processor.clustering import KMeans
from corelay.processor.embedding import TSNEEmbedding, UMAPEmbedding, EigenDecomposition
from corelay.processor.flow import Parallel, Sequential
import numpy as np
import torch
import matplotlib.pyplot as plt

# from run_ldce_baseline import get_dataset
from src.helpers.data_model_helpers import get_dataset
from cluster_variants import VARIANTS


def write_clustering_results(analysis_file_path, labels, eigenvalues, embedding, kmeans, umap, tsne):

    with h5py.File(analysis_file_path, 'a', locking=False) as analysis_file:

        # The name of the analysis is the name of the class
        analysis_name = 'layer and concept'# wordnet_id_map.get(class_index, f'{class_index:08d}')

        number_of_clusters_list = [5,7]    # [2,3,4,5,6,8,10]

        # Adds the indices of the samples in the current class to the analysis database
        analysis_group = analysis_file.require_group(analysis_name)
        # analysis_group['index'] = indices_of_samples_in_class.astype('uint32')

        analysis_group['index'] = np.arange(0, len(labels))

        # Adds the spectral embedding to the analysis database
        embedding_group = analysis_group.require_group('embedding')
        embedding_group['spectral'] = embedding.astype(np.float32)
        embedding_group['spectral'].attrs['eigenvalue'] = eigenvalues.astype(np.float32)

        # Adds the t-SNE embedding to the analysis database
        embedding_group['tsne'] = tsne.astype(np.float32)
        embedding_group['tsne'].attrs['embedding'] = 'spectral'
        embedding_group['tsne'].attrs['index'] = np.array([0, 1])

        # Adds the uMap embedding to the analysis database
        embedding_group['umap'] = umap.astype(np.float32)
        embedding_group['umap'].attrs['embedding'] = 'spectral'
        embedding_group['umap'].attrs['index'] = np.array([0, 1])


        cluster_group = analysis_group.require_group('cluster')
        # Add own agg
        # for md, clustering in zip(max_distance_list, dist_agg):
        #     clustering_dataset_name = f'dist_agg-{md}'
        #     cluster_group[clustering_dataset_name] = clustering
        #     cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
        #     cluster_group[clustering_dataset_name].attrs['index'] = np.arange(embedding.shape[1], dtype=np.uint32)
        # cluster_group['dist_agg'] = dist_agg
        # cluster_group['dist_agg'].attrs['embedding'] = 'spectral'
        # cluster_group['dist_agg'].attrs['index'] = np.arange(embedding.shape[1], dtype=np.uint32)

        # Adds the k-means clustering of the embeddings to the analysis database
        for number_of_clusters, clustering in zip(number_of_clusters_list, kmeans):
            clustering_dataset_name = f'kmeans-{number_of_clusters:02d}'
            cluster_group[clustering_dataset_name] = clustering
            cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
            cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
            cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
                embedding.shape[1],
                dtype=np.uint32
            )

        # # Adds the DBSCAN epsilon clustering of the embeddings to the analysis database
        # for number_of_clusters, clustering in zip(number_of_clusters_list, dbscan):
        #     clustering_dataset_name = f'dbscan-eps={number_of_clusters / 10.0:.1f}'
        #     cluster_group[clustering_dataset_name] = clustering
        #     cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
        #     cluster_group[clustering_dataset_name].attrs['index'] = numpy.arange(
        #         embedding.shape[1],
        #         dtype=numpy.uint32
        #     )

        # # Adds the HDBSCAN clustering of the embeddings to the analysis database
        # clustering_dataset_name = 'hdbscan'
        # cluster_group[clustering_dataset_name] = hdbscan
        # cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
        # cluster_group[clustering_dataset_name].attrs['index'] = numpy.arange(
        #     embedding.shape[1],
        #     dtype=numpy.uint32
        # )

        # Adds the Agglomerative clustering of the embeddings to the analysis database
        # for number_of_clusters, clustering in zip(number_of_clusters_list, agglomerative):
        #     clustering_dataset_name = f'agglomerative-{number_of_clusters:02d}'
        #     cluster_group[clustering_dataset_name] = clustering
        #     cluster_group[clustering_dataset_name].attrs['embedding'] = 'spectral'
        #     cluster_group[clustering_dataset_name].attrs['k'] = number_of_clusters
        #     cluster_group[clustering_dataset_name].attrs['index'] = np.arange(
        #         embedding.shape[1],
        #         dtype=np.uint32
        #     )

        # If the attributions were computed on the training split of the dataset, then the training flag is set
        # if train_flag is not None:
        #     cluster_group['train_split'] = train_flag

def get_clustering_pipeline(variant='spectral', number_of_neighbors=8, number_of_eigenvalues=8):


    pre_processing_pipeline = VARIANTS[variant]['preprocessing']
    distance_metric = VARIANTS[variant]['distance']

    number_of_clusters_list = [5, 7]    # [2,3,4,5,6,8,10]

    pipeline = SpectralClustering(
        # preprocessing=pre_processing_pipeline,
        pairwise_distance=distance_metric,
        affinity=SparseKNN(n_neighbors=number_of_neighbors, symmetric=True),
        embedding=EigenDecomposition(n_eigval=number_of_eigenvalues, is_output=True),
        clustering=Parallel([
            Parallel([
                KMeans(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
            ], broadcast=True),
            # Parallel([
            #     DBSCAN(eps=number_of_clusters / 10.0) for number_of_clusters in number_of_clusters_list
            # ], broadcast=True),
            # HDBSCAN(),
            # Parallel([
            #     AgglomerativeClustering(n_clusters=number_of_clusters) for number_of_clusters in number_of_clusters_list
            # ], broadcast=True),
            Parallel([
                UMAPEmbedding(),
                TSNEEmbedding(perplexity=10.),          # 0.6
            ], broadcast=True)
        ], broadcast=True, is_output=True)
    )

    return pipeline

def get_attribution_data(attribution_file_path, indices, param='attribution', layer='features.37'):
    """ Load attribution data from file for the provided indices and given parameter. """

    with h5py.File(attribution_file_path, 'r', locking=False) as attributions_file:

        # print(attributions_file.keys())
        # print(attributions_file[layer].keys())
        # print(indices)
        # print(attributions_file['label'].shape)
        if layer in attributions_file:
            attribution_data = attributions_file[layer][param][indices, :]
        else:
            attribution_data = attributions_file[param][layer][indices, :]
        labels = attributions_file['label'][indices]
        if 'train' in attributions_file:
            # train_flag = attributions_file['train'][indices_of_samples_in_class.tolist()]
            train_flag = attributions_file['train']#[indices_of_samples_in_class.tolist()]
        else:
            train_flag = None

        print(attribution_data.shape)

    return attribution_data, labels

def visualize_distribution(index, umap, tsne, labels, cmap="Accent"):

    unique_labels, unique_indices = np.unique(labels, return_inverse=True)

    plt.figure()
    scatter = plt.scatter(umap[:,0], umap[:,1], c=unique_indices, cmap=cmap)

    # plt.legend(['orig_class', 'cf_class', 'datapoint', 'counterfactual'])
    plt.legend(handles=scatter.legend_elements()[0], labels=['orig_class', 'cf_class', 'datapoint', 'counterfactual'])
    plt.savefig(f'/results/counterfactuals/clusters/umap_{index}.png')
    plt.close()

    plt.figure()
    scatter = plt.scatter(tsne[:,0], tsne[:,1], c=unique_indices, cmap=cmap)

    plt.legend(handles=scatter.legend_elements()[0], labels=['orig_class', 'cf_class', 'datapoint', 'counterfactual'])
    plt.savefig(f'/results/counterfactuals/clusters/tsne_{index}.png')
    plt.close()

def compute_clustering(index, analysis_file_path, attribution_data, labels):

    pipeline = get_clustering_pipeline(number_of_neighbors=16, number_of_eigenvalues=32)

    # Run clustering pipeline
    (eigenvalues, embedding), (kmeans, (umap, tsne)) = pipeline(attribution_data)

    # Write analysis file
    # write_clustering_results(analysis_file_path, labels, eigenvalues, embedding, kmeans, umap, tsne)
    # print('Successfully wrote results.')

    visualize_distribution(index, umap, tsne, labels)

def compare_sample_to_data_distributions(index, cfg, analysis_file_path, indices):

    # Get original, predicted and counterfactual class

    # get indices of samples (and labels)

    attributions, labels = get_attribution_data(cfg.attribution_file_path, indices, layer=cfg.concept_layer, param='attribution')
    print(f'Attributions type: {type(attributions)} with shape {attributions.shape}')

    # Load original CF datapoint
    cf_attr, cf_label = get_attribution_data(cfg.cf_attr_file_path, index, layer=cfg.concept_layer, param='attribution')
    # print(cf_attr.shape)

    # Load generated CF datapoint
    cf_gen_attr, cf_gen_label = get_attribution_data(cfg.cf_gen_attr_file_path, index, layer=cfg.concept_layer, param='attribution')

    # add sample point and/or counterfactual to attribution data
    attribution_data = np.concatenate([attributions, cf_attr[None, :]], axis=0)
    labels = np.concatenate([labels, [1500]], axis=0)

    attribution_data = np.concatenate([attribution_data, cf_gen_attr[None, :]], axis=0)
    labels = np.concatenate([labels, [2000]], axis=0)

    # compute clustering
    compute_clustering(index, analysis_file_path, attribution_data, labels)


# @hydra.main(version_base=None, config_path="../configs/clustering", config_name="v1")
def run_cluster_evaluation(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load baseline dataset, e.g. all other evaluation data
    base_dataset = get_dataset(cfg, last_data_idx=0, base=True)

    cf_dataset = get_dataset(cfg, last_data_idx=0, base=False)
    cf_dataset = torch.utils.data.Subset(cf_dataset, np.arange(50))

    data_loader = torch.utils.data.DataLoader(cf_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, batch in enumerate(data_loader):

        if "return_tgt_cls" in cfg.data and cfg.data.return_tgt_cls:
            image, label, tgt_classes, unique_data_idx = batch
            # tgt_classes = tgt_classes.to(device)

        print(f'{label} -> {tgt_classes}')

        base_labels = base_dataset.targets
        base_labels = torch.Tensor([int(bl) for bl in base_labels])


        orig_class_inds = (base_labels == label.item()).nonzero(as_tuple=True)[0]
        cf_class_inds = (base_labels == tgt_classes.item()).nonzero(as_tuple=True)[0]

        # print(orig_class_inds)
        # print(cf_class_inds)

        indices = torch.cat([orig_class_inds, cf_class_inds], dim=0).numpy()
        indices.sort()

        # print(indices)

        analysis_file_path = os.path.splitext(cfg.analysis_file_path)
        analysis_file_path = "".join([analysis_file_path[0], str(i), analysis_file_path[1]])
        # print(analysis_file_path)

        compare_sample_to_data_distributions(i, cfg, analysis_file_path, indices)


        # compare_sample_to_data_distributions()

    # load samples for counterfactual computation

    # ( encode data ? )


@hydra.main(version_base=None, config_path="../../configs/clustering", config_name="v1")
def main(cfg) -> None:

    # compare_classes(cfg)
    run_cluster_evaluation(cfg)


if __name__ == '__main__':
    main()
