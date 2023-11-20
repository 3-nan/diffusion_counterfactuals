""" Script with functionality for extracting relevant concepts. """

def get_latent_representation(classifier, sample, layer):

    return NotImplementedError

def extract_differing_concepts(repr, cf_repr):

    return NotImplementedError

def compute_concept_conditioning(classifier, sample, layer, orig_class, cf_class):
    """ Compute latent space representations for actual and counterfactual class.
        Compare representations and extract most important and differing concepts.
    """

    # Explain classifier
    repr = get_latent_representation(classifier, sample, layer, orig_class)

    # Explain classifier for counterfactual class
    cf_repr = get_latent_representation(classifier, sample, layer, cf_class)

    # Extract concepts
    conditions = extract_differing_concepts(repr, cf_repr)

    return conditions
