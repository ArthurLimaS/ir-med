import etl_functions as etl
# import math
import numpy as np
from jaro import jaro_winkler_metric
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def template():
    """
    Breve descrição da função
    
    Parameters:
    ---------
    {parameter_name} : {parameter_type}[, default ?]
        {Parameter description}

    {parameter_name} : {parameter_type}[, default ?]
        {Parameter description}

    Returns:
    ---------
    {return_type}
        {Return description}
    """
    return -1

def extract_relevant_words(df_cmed, columns, verbose = False):
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

# Extracts from the column 'desc' of a notice the words that appear in the CMED report
def sep_desc(desc, cmed_ai_words, cmed_pr_words):
    desc_ai = ""
    desc_pr = ""

    for tok in word_tokenize(desc):
        if tok in cmed_ai_words:
            desc_ai += tok + " "
        
        if tok in cmed_pr_words:
            desc_pr += tok + " "
        
    return desc_ai, desc_pr



# Macro function that runs the medicine identification process
def predict(df_cmed, grouped_cmed, desc_ai, desc_pr, und):

    # Classification of the active_ingredient
    active_ingredient, _ = match_ai(grouped_cmed, desc_ai)

    # Coleta dos medicamentos da CMED que possuem o principio ativo apontado
    df_cmed_filtered = df_cmed.iloc[grouped_cmed[grouped_cmed['key'] == active_ingredient].reset_index()['indexes'][0]]
    
    return filter_prs(df_cmed_filtered, desc_ai, desc_pr, und, active_ingredient)



# Function that predicits the pharmaceutical ingredient
def match_ai(grouped_cmed, desc_ai):
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



# Function that returns the presentations that have the most intersection with desc_pr
def filter_prs(df_cmed_filtered, desc_ai, desc_pr, und, active_ingredient):

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