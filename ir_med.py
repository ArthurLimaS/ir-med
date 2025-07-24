import etl_functions as etl
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
    Runs the complete medicine identification process based on a description
    from a public notice.

    Parameters
    ----------
    df_cmed : pandas.DataFrame
        DataFrame containing the CMED dataset.

    grouped_cmed : pandas.DataFrame
        DataFrame containing CMED data grouped by active ingredient.

    desc_ai : str
        Substring of the public notice description related to the active
        ingredient.

    desc_pr : str
        Substring of the public notice description related to the pharmaceutical
        presentation.

    und : str
        Unit of measurement specified in the public notice.

    Returns
    -------
    tuple
        A tuple containing:
        
        - list of int: Indices of the best matching presentations in the CMED data.
        - dict: Metadata about the identification process, including:
            - 'active_ingredient_found' (str): The matched active ingredient.
            - 'desc_ai' (str): Active ingredient description used for matching.
            - 'similarity_value' (float): Similarity score of the best active ingredient match.
            - 'desc_pr' (str): Pharmaceutical presentation description used for matching.
            - 'len_cmed_filtered' (int): Number of rows in the filtered CMED DataFrame.
            - 'len_best_matches' (int): Number of matching presentations after filtering.
            - 'pct_set_reduction' (float): Percentage reduction in CMED entries after filtering.
    """

    # Classification of the active_ingredient
    ai_found, predict_ai_metadata = predict_ai(grouped_cmed, desc_ai)

    # Colect medicines that have the active ingredient
    idxs = grouped_cmed.loc[grouped_cmed['key'] == ai_found, 'indexes']
    df_cmed_filtered = df_cmed.iloc[idxs.values[0]]

    # Filter medicines based on the pharmaceutical presentation
    best_matches, match_presentations_metadata = match_presentations(df_cmed_filtered,
                                                                     desc_pr,
                                                                     und)
    
    # Create process metadata for the whole matching process
    process_metadata = {'active_ingredient_found': ai_found}
    process_metadata.update(predict_ai_metadata)
    process_metadata.update(match_presentations_metadata)

    return best_matches, process_metadata

def predict_ai(grouped_cmed, desc_ai):
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
            - 'desc_ai' (str): Active ingredient description used for matching.
            - 'similarity_value' (float): Similarity score of the best active ingredient match.
    """

    # Sort the description alphabetically
    desc_ai_sorted = etl.sort_alphabetically(desc_ai)

    # Calculate the Jaro-Winkler similarity metric for each key in the grouped CMED data
    metrics = grouped_cmed['key_sorted'] \
              .map(lambda x: jaro_winkler_metric(desc_ai_sorted, x))
    
    # Find the index of the best match
    best_idx = metrics.idxmax()

    # Get the best match key
    best_match_key = grouped_cmed['key'][best_idx]
    best_match_value = metrics[best_idx]

    process_metadata = {'desc_ai': desc_ai,
                        'similarity_value': best_match_value}

    return best_match_key, process_metadata

def match_presentations(df_cmed_filtered, desc_pr, und):
    """
    Identifies entries in the CMED DataFrame whose pharmaceutical presentations 
    best match the presentation description provided in the public notice.

    This function operates on CMED data already filtered by an active
    ingredient.
    
    Parameters:
    ---------
    df_cmed_filtered : DataFrame
        DataFrame containing the filtered CMED data based on the active
        ingredient.
    
    desc_pr : str
        Substring of the public notice description related to the pharmaceutical
        presentation.
    
    und : str
        Unit of measurement specified in the public notice.

    Returns:
    ---------
    tuple
        A tuple containing:
        - list of int: Indices of the best matching presentations in the CMED data.
        - dict: Metadata about the identification process, including:
            - 'desc_pr' (str): Pharmaceutical presentation description used for matching.
            - 'len_cmed_filtered' (int): Number of rows in the filtered CMED DataFrame.
            - 'len_best_matches' (int): Number of matching presentations after filtering.
            - 'pct_set_reduction' (float): Percentage reduction in CMED entries after filtering.
    """

    sets = extract_ngrams(desc_pr) + extract_ngrams(und)
    best_count = 0
    best_matchs = []

    # Pre-tokenize presentations in the CMED DataFrame
    presentations_tokenized = df_cmed_filtered['apresentacao'] \
                              .map(word_tokenize)

    for idx_cmed, tokens_cmed in presentations_tokenized.items():
        count = 0

        for st in sets:            
            # If set only has one token, check if that tokens appears in the
            # CMED presentation
            if len(st) == 1:
                if st[0] in tokens_cmed:
                    count += 1

            # If set has more than a token, check if the sequence of tokens
            # appears, in that order, in the CMED presentation
            else:
                window_range = len(tokens_cmed) - len(st) + 1

                for i in range(window_range):
                    if tokens_cmed[i:i+len(st)] == st:
                        count += 1
                        break

        if count > best_count:
            best_count = count
            best_matchs = [idx_cmed]

        elif count == best_count:
            best_matchs.append(idx_cmed)

    process_metadata = {
        'desc_pr': desc_pr,
        'quant_presentations_matched': len(best_matchs),
        'size_cmed_filtered': len(df_cmed_filtered),
        'pct_set_reduction': (1 - (len(best_matchs) / len(df_cmed_filtered))) \
                             if len(df_cmed_filtered) else 1.0
    }

    return best_matchs, process_metadata

def extract_ngrams(desc_pr):
    """
    Extracts all contiguous sequences of words (n-grams) from a given
    pharmaceutical presentation description.

    Parameters
    ----------
    desc_pr : str
        Substring of the public notice description related to the pharmaceutical
        presentation.

    Returns
    -------
    list of list of str
        A list where each sublist represents a contiguous sequence of words
        (n-gram) extracted from the input description.
    """
    tokens = word_tokenize(desc_pr)
    n_tokens = len(tokens)

    sets = [tokens[start_index : start_index + slice_range] \
            for slice_range in range(1, n_tokens + 1) \
                for start_index in range(n_tokens - slice_range + 1)]

    return sets