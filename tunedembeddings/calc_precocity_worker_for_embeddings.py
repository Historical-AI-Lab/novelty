import pandas as pd
import numpy as np
import random, sys, os, math
from scipy.stats import entropy
from collections import Counter
import warnings

warnings.filterwarnings('error')

def get_vectors(paperId, data, function_string, chunksfordoc):
    papervectors = []
    if function_string == 'cosine':
        for i in range (0, 10000):
            chunkid = paperId + '-' + str(i)
            if i > 9995:
                print('Dangerously long document.')
            if chunkid not in data:        
                break
            else:
                papervectors.append((chunkid, data[chunkid]))
    else:
        if paperId not in chunksfordoc:
            return papervectors
        else:
            for chunk in chunksfordoc[paperId]:
                papervectors.append((chunk, data[chunk]))

    return papervectors

def any_overlap(a, b):
    if len(a.intersection(b)) > 0:
        return True
    else:
        return False

def z_transform(a_cosine):
    try:
        z_transformed = 0.5 * np.log((1 + a_cosine) / (1 - a_cosine))
    except:
        print(a_cosine, flush = True)
        z_transformed = 0

    return z_transformed

def getallchunks(paperId_list, chunksfordoc):
    """
    Retrieves all chunks for the given paper IDs.

    Args:
        paperId_list (list): A list of paper IDs.
        chunksfordoc (dict): A dictionary containing chunks for each document.

    Returns:
        set: A set of all chunks.

    """
    allchunks = set()
    for paperId in paperId_list:
        if paperId in chunksfordoc:
            allchunks.update(chunksfordoc[paperId])
               
    return allchunks

def get_exclusions(paperId, exclusions, chunksfordoc):
    
    '''
    Accepts a paperId, a dict of exclusions, and a dict that maps
    from docids to lists of chunkids. Returns a dict where keys
    are filter_states and values are sets of chunkids that are
    excluded from analysis for that filter_state. Note that the
    sets have a matryoshka structure: if a chunk is excluded
    for 'train' it's also excluded for 'trainauth', and a chunk
    excluded from 'trainauth' will also be excluded
    from 'trainauthquote.'
    '''

    exclude_for_this = dict()
    exclude_for_this['no filter'] = set()
    exclude_for_this['train'] = getallchunks(exclusions['train'], chunksfordoc)
    if paperId in exclusions:
        exclude_for_this['trainauth'] = exclude_for_this['train'].union(getallchunks(exclusions[paperId]['author overlaps'], chunksfordoc))
        exclude_for_this['trainauthquote'] = exclude_for_this['trainauth'].union(exclusions[paperId]['chunks that quote'])
    else:
        exclude_for_this['trainauth'] = set()
        exclude_for_this['trainauthquote'] = set()
    
    return exclude_for_this
     
def calculate_a_year(package):
    '''
    Calculates Kullback-Leibler divergence forward and back in time
    for a single year in the metadata frame.

    Accepts as its argument a 5-tuple that should be:
    
    1 centerdate        an integer
    2 meta              metadata DataFrame
    3 data              a dict where keys are chunkids and values are vectors
    4 exclusions        a dict where keys are paperIds and values are dicts
                        with keys 'author overlaps' and 'chunks that quote'
                        "author overlaps" is a set of paperIds
                        "chunks that quote" is a set of chunkids
                        there is in addition one special key that instead of a paperId
                        is 'train', and it contains a set of chunkids that are to be excluded
                        for all filter_conditions that contain 'train'
    5 function_string   a string that tells us whether to use "kld" or "cosine"

    Returns two objects:

    1 summary statistics listing the novelty, transience, and resonance
    averaged over different timespans and fractions. For def of novelty,
    transience and resonance see Barron et al. (2018).

    2 cosine-distance calculated backward and forward in time. This is not
    something we're necessarily planning to use in the experiment, but
    it's here to permit various kinds of sanity checking.
    '''
    errors = Counter()
    centerdate, meta, data, exclusions, function_string = package

    fractions2check = [1.0, 0.1, 0.05]

    filter_states = ['no filter', 'train', 'trainauth', 'trainauthquote']
    
    # Overall the "filtered" variable has four conditions:
    # nofilter -- no filtering at all everything included
    # train -- only training corpus excluded
    # trainauth -- training corpus and author overlaps excluded
    # trainauthquote -- training corpus, author overlaps, and cites/quotes excluded

    timeradii2check = [-20, -10, 10, 20]

    positive_radii = [10, 20]

    aggregates2check = [1.0, 0.25]

    condition_package = (fractions2check, filter_states, positive_radii, aggregates2check)

    # We have a challenge to handle. Our exclusions are defined to the smaller
    # 512-token chunks, but the topic model combines those chunks to produce 
    # larger ones of *at least* 512 words. So we need a map from the topic chunks
    # to the exclusions

    chunksfordoc = dict()

    chunk_mapper = dict()
    if function_string == 'kld' or function_string == 'cosine':
        for chunkid, vec in data.items():
            chunkindexes = [x for x in chunkid.split('-')[1].split('.')]
            docid = chunkid.split('-')[0]
            if docid not in chunksfordoc:
                chunksfordoc[docid] = []
            chunksfordoc[docid].append(chunkid)
            for idx in chunkindexes:
                equivalent_chunk = docid + '-' + idx
                chunk_mapper[equivalent_chunk] = chunkid

    print(len(chunk_mapper))

    databyyear = dict()

    ctr = 0

    numbers_of_exclusions = []
    numbers_of_overlaps= []

    paperstocheck = meta.index[meta.year == centerdate].tolist()

    print(len(paperstocheck), ' papers to check.', flush = True)

    doc_precocities = dict() 

    for paperId in paperstocheck:

        ctr += 1
        
        if ctr % 20 == 2:
            print(centerdate, ctr, flush=True)

        papervectors = get_vectors(paperId, data, function_string, chunksfordoc)

        if len(papervectors) == 0:
            print(paperId, ' not found')
            continue
        else:
            number_of_chunks = len(papervectors)

        
        exclude_for_this = get_exclusions(paperId, exclusions, chunksfordoc)
        
        # exclude_for_this is a dict where keys are filter_states, and values
        # are sets of chunkids that are forbidden for that filter state

        distances = dict()    
        # note these include both novelties and transiences
        # the positive or negative character of the comp_date
        # is what tells you which are which


        for chunk_num in range(number_of_chunks):
            for filtered in filter_states:
                for comp_date in range (-20, 21):
                    if comp_date == 0:
                        pass
                    else:
                        distances[(chunk_num, filtered, comp_date)] = []

        for comp_date in range(-20, 21):
            if comp_date == 0:
                continue
            comps = meta.index[meta.year == centerdate + comp_date].tolist()
            for comp_paper in comps:

                comp_vectors = get_vectors(comp_paper, data, function_string, chunksfordoc)

                for c_idx, chunktuple in enumerate(comp_vectors):    # c_idx is not actually used
                    chunkid, c_vec = chunktuple
                    
                    for p_idx, papertuple in enumerate(papervectors):
                        paperchunkid, p_vec = papertuple             # paperchunkid is not actually used
                        if function_string == 'kld':
                            distance = entropy(p_vec, c_vec)
                            # always surprise of the paper relative to comparison
                        else:
                            try:
                                dotproduct = np.dot(p_vec, c_vec)  # this can range from -1 to 1
                                distance = -1 * (z_transform(dotproduct)) # now it ranges from -inf to +inf
                                # not properly a distance, but we're using it as one
                            except RuntimeWarning as e:
                                print(e, paperId, comp_paper, flush = True)
                                pass 

                        if distance == 0:
                            print('zero distance', paperId, comp_paper, flush = True)

                        for filtered in filter_states:

                            if chunkid in exclude_for_this[filtered]:
                                pass 
                            else:
                                distances[(p_idx, filtered, comp_date)].append(distance)

        novelties = dict()
        for p_idx in range(number_of_chunks):
            for fraction in fractions2check:
                for radius in timeradii2check:
                    for filtered in filter_states:
                        novelties[(p_idx, fraction, filtered, radius)] = []

        for p_idx in range(number_of_chunks):
            for fraction in fractions2check:
                for filtered in filter_states:
                    for comp_date in range(-20, 21):
                        if comp_date == 0:
                            continue
                        sorted_distances = sorted(distances[(p_idx, filtered, comp_date)])
                        number_to_average = math.ceil(len(sorted_distances) * fraction)
                        if number_to_average < 1:
                            number_to_average = 1
                            errors['numavg'] += 1
                            print('empty set', comp_date)
                        the_mean = np.mean(sorted_distances[0: number_to_average])
                        for radius in timeradii2check:
                            if abs(comp_date) <= abs(radius) and (np.sign(radius) == np.sign(comp_date)):
                                novelties[(p_idx, fraction, filtered, radius)].append(the_mean)

        try:
            precocities = dict()
            for p_idx in range(number_of_chunks):
                for fraction in fractions2check:
                    for filtered in filter_states:
                        for pos_radius in positive_radii:
                            novelty = np.mean(novelties[(p_idx, fraction, filtered, -pos_radius)])
                            transience = np.mean(novelties[(p_idx, fraction, filtered, pos_radius)])
                            precocity = novelty - transience
                            precocities[(p_idx, fraction, filtered, pos_radius)] = (precocity, novelty, transience)
        except RuntimeWarning as e:
            print(novelties)
            print(fraction, filtered, pos_radius)
            print(e)
            break

        document_precocity = dict()
        for fraction in fractions2check:
                for filtered in filter_states:
                    for pos_radius in positive_radii:
                        chunk_precocs = []
                        for p_idx in range(number_of_chunks):
                            chunk_precocs.append(precocities[(p_idx, fraction, filtered, pos_radius)])

                        chunk_precocs = sorted(chunk_precocs, reverse = True)
                        # note, since precocity is the first item in the tuple this will sort by precocity

                        for fraction_to_aggregate in aggregates2check:
                            number_to_average = math.ceil(len(chunk_precocs) * fraction_to_aggregate)
                            mean_precocity = np.mean([x[0] for x in chunk_precocs[0: number_to_average]])
                            mean_novelty = np.mean([x[1] for x in chunk_precocs[0: number_to_average]])
                            mean_transience = np.mean([x[2] for x in chunk_precocs[0: number_to_average]])
                            document_precocity[(fraction, filtered, pos_radius, fraction_to_aggregate)] =\
                            (mean_precocity, mean_novelty, mean_transience)

        document_precocity['num_chunks'] = number_of_chunks
        doc_precocities[paperId] = document_precocity

    print(errors)
    return doc_precocities, centerdate, condition_package

