""" Module for providing different generation targets. """

TARGET_MODES = [
    'target_class',
    'ref_sample',
    'ref_latent'
]

def target_selection(encodings_file, mode='target_class', feat_method="relevance", layer_name):
    """ Main target selection function. """
    
    assert mode in TARGET_MODES

    # Options:
    # A: class based
    # B: reference sample based
    #   1: single ref sample
    #   2: multiple samples
    # C: latent representation
    #
    # A: gradient
    # B: relevance
    # C: activation

    # Params:
    # num_samples to consider?
    # num_concepts
    # concept_ids?

    if mode=='target_class':
        print(f"Mode {mode} selected.")

        # default from file (wordnet) -> tgt_classes

        # class of near miss/cluster (?)

    elif mode == 'ref_sample':
        print(f"Mode {mode} selected.")

        latent_sample = encode_sample(model, sample, layer_name, label)

        # Near Miss
        near_misses = get_near_miss(data_base, latent_sample, layer_name, k=1)

        # Near Cluster
        # near_cluster = get_near_cluster(latent_sample, k=1)

        # Selected samples
