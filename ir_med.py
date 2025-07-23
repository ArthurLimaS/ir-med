import etl_functions as etl
# import math
import numpy as np
from jaro import jaro_winkler_metric
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def identify_relevant_words(df_cmed, columns, verbose = False):
    """
    Extracts tokens (words) from the specified columns of a DataFrame.

    For each selected column, returns a list of all unique tokens found in that
    column.

    Parameters
    ----------
    df_cmed : DataFrame
        DataFrame containing the CMED data.

    columns : list of str
        Names of the columns to analyze.

    verbose : bool, default False
        Whether to display progress information during processing.

    Returns
    -------
    dict
        A dictionary where each key is a column name and each value is a list of
        unique tokens found in that column.
    """
    
    answer_dict = {}
    iterator = columns
    if verbose:
        iterator = tqdm(columns)

    for col in iterator:
        # Concatena todas as strings da coluna, tokeniza e pega os únicos
        all_text = " ".join(df_cmed[col].astype(str))
        tokens = word_tokenize(all_text)
        answer_dict[col] = np.unique(tokens)

    return answer_dict

def split_description(desc, cmed_ai_words, cmed_pr_words):
    """
    Extract words from a medicine description that match tokens identified in
    the CMED dataset.

    Parameters
    ----------
    desc : str
        Description of a medicine from the public notice.

    cmed_ai_words : list of str
        List of tokens identified as active ingredients in the CMED dataset.

    cmed_pr_words : list of str
        List of tokens identified as pharmaceutical presentations in the CMED
        dataset.

    Returns
    -------
    tuple of str
        A tuple containing:
        - A string with the words from the description that match active ingredient terms.
        - A string with the words from the description that match pharmaceutical presentation terms.
    """

    tokens = word_tokenize(desc)
    desc_ai = " ".join([tok for tok in tokens if tok in cmed_ai_words])
    desc_pr = " ".join([tok for tok in tokens if tok in cmed_pr_words])
        
    return desc_ai, desc_pr

def predict(df_cmed, grouped_cmed, desc_ai, desc_pr, und):
    """
    Run the complete medicine identification process based on a description from
    a public notice.

    Parameters
    ----------
    df_cmed : DataFrame
        DataFrame containing the CMED dataset.

    grouped_cmed : DataFrame
        DataFrame containing grouped CMED data.

    desc_ai : str
        Portion of the description related to the active ingredient.

    desc_pr : str
        Portion of the description related to the pharmaceutical presentation.

    und : str
        Unit of measurement specified in the public notice.

    Returns
    -------
    tuple
        A tuple containing:
        - list of int: Indices of the best matching presentations in the CMED data.
        - dict: Metadata about the identification process, including:
            - 'desc_ai': Active ingredient description used for matching.
            - 'active_ingredient_found': The matched active ingredient.
            - 'similarity_value': Similarity score of the best active ingredient match.
            - 'desc_pr': Pharmaceutical presentation description used for matching.
            - 'size_cmed_filtered': Number of rows in the filtered CMED DataFrame.
            - 'quant_presentations_matched': Number of matching presentations.
            - 'pct_set_reduction': Percentage reduction in CMED entries after filtering.
    """

    # Classification of the active_ingredient
    ai_found, match_ai_metadata = match_ai(grouped_cmed, desc_ai)

    # Colect medicines that have the active ingredient
    idxs = grouped_cmed.loc[grouped_cmed['key'] == ai_found, 'indexes']
    df_cmed_filtered = df_cmed.iloc[idxs]

    # Filter medicines based on the pharmaceutical presentation
    best_matches, filter_prs_metadata = filter_prs(df_cmed_filtered, desc_ai,
                                                 desc_pr, und, ai_found)
    
    # Create process metadata for the whole matching process
    process_metadata = match_ai_metadata
    process_metadata.update(filter_prs_metadata)

    return best_matches, process_metadata

def match_ai(grouped_cmed, desc_ai):
    """
    Predicts the most likely pharmaceutical active ingredient based on a given
    description.

    Parameters
    ----------
    grouped_cmed : DataFrame
        DataFrame containing grouped CMED data.

    desc_ai : str
        Description related to the active ingredient from the public notice.

    Returns
    -------
    tuple
        A tuple containing:
        - str: The best matching active ingredient key.
        - dict: Metadata about the matching process, including:
            - 'desc_ai': The description used for matching.
            - 'similarity_value': Similarity score of the best match.
    """

    desc_ai = etl.sort_alphabetically(desc_ai)
    best_match = -1
    best_match_key = ""

    for i in range(len(grouped_cmed['key_sorted'])):
        key_sorted = grouped_cmed['key_sorted'][i]

        # Similarity calculation
        metric = jaro_winkler_metric(desc_ai, key_sorted)

        if metric >= best_match:
            best_match = metric
            best_match_key = grouped_cmed['key'][i]

    process_metadata = {'desc_ai': desc_ai,
                        'similarity_value': best_match}

    return (best_match_key, process_metadata)

def filter_prs(df_cmed_filtered, desc_ai, desc_pr, und, active_ingredient):
    """
    Function that returns the presentations that have the most intersection with
    desc_pr
    
    Parameters:
    ---------
    df_cmed_filtered : DataFrame
        DataFrame containing the filtered CMED data based on the active ingredient.

    desc_ai : str
        Description related to the active ingredient from the public notice.
    
    desc_pr : str
        Description related to the pharmaceutical presentation from the public
        notice.
    
    und : str
        Unit description from the 'unidade' column in the notice data.
    
    active_ingredient : str
        The active ingredient that was matched from the CMED data.

    Returns:
    ---------
    tuple
        A tuple containing:
        - list of int: Indices of the best matching presentations in the CMED data.
        - dict: Metadata about the identification process, including:
            - 'desc_ai': Active ingredient description used for matching.
            - 'desc_pr': Pharmaceutical presentation description used for matching.
            - 'active_ingredient_found': The active ingredient that was matched.
            - 'size_cmed_filtered': Number of rows in the filtered CMED DataFrame.
            - 'quant_presentations_matched': Number of matching presentations.
            - 'pct_set_reduction': Percentage reduction in the CMED dataset after filtering.
    """

    sets = get_sets_from_desc_pr(desc_pr)
    und_sets = get_sets_from_desc_pr(und)
    sets.extend(und_sets)

    best_count = 0
    best_matchs = []

    for idx_cmed, row_cmed in df_cmed_filtered.iterrows():

        # Check if the presetation has the tokens found in the notice entry
        count = 0
        tokens_cmed = word_tokenize(row_cmed['apresentacao'])

        for st in sets:
            
            # If set only has one token, check if that tokens appears in the CMED presentation
            if len(st) == 1:
                if st[0] in tokens_cmed:
                    count += 1

            # If set has more than a token, check if the sequence of tokens appears, in that order, in the CMED presentation
            else:
                if st[0] in tokens_cmed:
                    check = True
                    initial_index = tokens_cmed.index(st[0])
                    
                    for i in range(1, len(st)):
                        current_index = (i+initial_index)
                        
                        if (current_index >= len(tokens_cmed)) or \
                            (st[i] != tokens_cmed[current_index]):

                            check = False
                            break
                                
                    if check:
                        count += 1

        if count > best_count:
            best_count = count
            best_matchs = [idx_cmed]
        elif count == best_count:
            best_matchs.append(idx_cmed)

    process_metadata = {'desc_ai': desc_ai,
                        'desc_pr': desc_pr,
                        'active_ingredient_found': active_ingredient,
                        'quant_presentations_matched': len(best_matchs),
                        'size_cmed_filtered': len(df_cmed_filtered),
                        'pct_set_reduction': (1 - (len(best_matchs) / len(df_cmed_filtered)))}

    return (best_matchs, process_metadata)

def get_sets_from_desc_pr(desc_pr):
    """
    Generates all possible sets of words from a given description.

    Parameters
    ----------
    desc_pr : str
        Description related to the pharmaceutical presentation from the public
        notice.

    Returns
    -------
    list of list of str
        A list containing all possible combinations of words extracted from the
        description.
    """
    tokens = word_tokenize(desc_pr)
    n_tokens = len(tokens)

    func = lambda x : (x**2 + x) / 2    # Calculate the number of sets to create
    quant_verificacoes = int(func(n_tokens))

    sets = []
    subset_size = 1
    reduction_value = 0

    for i in range(quant_verificacoes):
        x = i - reduction_value
        
        match = (n_tokens - subset_size + 1)

        if x >= match:
            x -= (n_tokens - subset_size + 1)
            reduction_value += (n_tokens - subset_size + 1)
            subset_size += 1

        st = []
        for j in range(subset_size):
            st.append(tokens[x+j])

        sets.append(st)

    return sets