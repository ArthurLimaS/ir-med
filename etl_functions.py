import nltk
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from unidecode import unidecode

def load_cmed(path, sep = ';'):
    """
    Load the CMED dataset from a .csv file
    
    Parameters
    ---------
    path : str, path object or file-like object
        Path to a CSV file containing the CMED table. Can be a string, a
        PathLike object, or a file-like object with a ``read()`` method.

    sep : str, default ';'
        Character used to separate fields in the CSV file.

    Returns
    ---------
    DataFrame
        A DataFrame containing the data from the CMED file.
    """

    return pd.read_csv(path, sep = sep)

def std_cols_names_preprocessing(df_cmed):
    """
    Standardizes column names by converting them to lowercase, removing
    accents, and replacing spaces with underscores.

    Parameters
    ----------
    df_cmed : DataFrame
        A DataFrame containing the data from the CMED file.

    Returns
    -------
    DataFrame
        A DataFrame with standardized column names.
    """

    # Copy the DataFrame
    new_df = df_cmed.copy()

    # Turn to lowercase + remove accents + change blank spaces for "_"
    new_df.columns = [
        unidecode(col.lower()).replace(' ', '_')
        for col in new_df.columns
    ]
    
    return new_df

def std_preprocessing(text, correct_ai = False, rem_nums = False,
                      rem_stopwords_ai = False, rem_stopwords_pr = False,
                      abbreviate_prs = True, rem_rep_tokens = False):
    """
    Applies a standard preprocessing pipeline to text from CMED and Public
    Notices data.

    The complete pipeline includes the following steps:
    1. Convert text to lowercase.
    2. Remove accentuation.
    3. Remove special characters (i.e., characters not in the ranges a–z, A–Z, 0–9, including underscores).
    4. Remove URLs starting with "http".
    5. Remove URLs starting with "www".
    6. Correct misspellings of pharmaceutical ingredients. [Optional, default: False]
    7. Insert spaces between numbers and letters.
    8. Remove numbers. [Optional, default: False]
    9. Remove words that hinder the identification of pharmaceutical ingredients. [Optional, default: False]
    10. Remove words that hinder the identification of pharmaceutical presentations. [Optional, default: False]
    11. Abbreviate presentation terms based on the ANVISA vocabulary. [Optional, default: True]
    12. Remove repeated words. [Optional, default: False]

    Parameters
    ----------
    text : str
        Text to be preprocessed.

    correct_ai : bool, default False
        Whether to correct misspellings of pharmaceutical ingredients. The
        correction rules are defined in the ``CORRECTION`` dictionary.

    rem_nums : bool, default False
        Whether to remove numbers from the text.

    rem_stopwords_ai : bool, default False
        Whether to remove words that hinder the identification of pharmaceutical
        active ingredients. These words are listed in the ``STOPWORDS_AI``
        dictionary.

    rem_stopwords_pr : bool, default False
        Whether to remove words that hinder the identification of pharmaceutical
        presentations. These words are listed in the ``STOPWORDS_PR``
        dictionary.

    abreviate_prs : bool, default True
        Whether to abbreviate presentation-related terms based on the ANVISA
        vocabulary. Abbreviations are defined in the ``ANVISA_ABBREVIATOR``
        dictionary.

    rem_rep_tokens : bool, default False
        Whether to remove repeated tokens (duplicate words) from the text.

    Returns
    -------
    str
        The preprocessed text.
    """

    # Lowercase and remove accents
    text = unidecode(text.lower())
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters (keep only words and numbers)
    text = re.sub('\W',' ', text)         # Removes specials characters and leaves only words

    tokens = word_tokenize(text)

    # Correct incorrect writing of pharmaceutical ingredients
    if correct_ai:
        CORRECTIONS = {'acilovir': 'aciclovir',
                        'amoxilina': 'amoxicilina',
                        'benzoilmetronidazol': 'metronidazol',
                        'cabidopa': 'carbidopa',
                        'carvedilo': 'carvedilol',
                        'cetamina': 'escetamina',
                        'clonazepan': 'clonazepam',
                        'deslanosido': 'deslanosideo',
                        'dexamatesona': 'dexametasona',
                        'dexametasoma': 'dexametasona',
                        'dexclorfemiramina': 'dexclorfeniramina',
                        'dexclofeniramina': 'dexclorfeniramina',
                        'dextrocetamina': 'escetamina',
                        'dimenitrato': 'dimenidrinato',
                        'diporina': 'dipirona',
                        'dolantina': 'petidina',
                        'enoxoparina': 'enoxaparina',
                        'espirolactona': 'espironolactona',
                        'estrogeno': 'estrogenios',
                        'estrogenos': 'estrogenios',
                        'folinico': 'folico',
                        'fomoterol': 'formoterol',
                        'hidroclotiazida':'hidroclorotiazida',
                        'hidrocortizona': 'hidrocortisona',
                        'halpperidol': 'haloperidol',
                        'kcl': 'potassio',
                        'meloxican': 'meloxicam',
                        'meropnem': 'meropenem',
                        'metoclopamida': 'metoclopramida',
                        'metroninazol': 'metronidazol',
                        'midazolan': 'midazolam',
                        'nacl': 'sodio',
                        'ondasetrona': 'ondansetrona',
                        'oxcarbamazepin': 'oxcarbazepina',
                        'oxcarbamazepina': 'oxcarbazepina',
                        'oxitocina': 'ocitocina',
                        'piperaciclina': 'piperacilina',
                        'subactant': 'sulbactam',
                        'sulfametazol': 'sulfametoxazol',
                        'tenoxican': 'tenoxicam',
                        'trimetroprima': 'trimetoprima'}
        
        tokens = [CORRECTIONS.get(tok, tok) for tok in tokens]

    # Insert blank space between numbers and words
    new_tokens = []
    for tok in tokens:
        # Split tokens into words and numbers
        split_tok = re.findall(r'[A-Za-z]+|\d+', tok)

        new_tokens.extend(split_tok)
    tokens = new_tokens

    # Remove numbers
    if rem_nums:
        tokens = [tok for tok in tokens if not tok.isdigit()]

    # Remove words that hinder the identification of pharmaceutical active ingredients
    if rem_stopwords_ai:
        STOPWORDS_AI = ['a', 'acetato', 'acido', 'anidra',
                        'benzatina', 'besilato', 'bicarbonato', 'bidestilada','bissulfato', 'brometo', 'bromidrato', 'bultiprometo',
                        'c', 'calcica', 'carbonato', 'citrato', 'clavulanato', 'cloreto', 'cloridrato', 'com', 'complexo',
                        'd', 'da', 'de', 'di', 'dicloridrato', 'diidratada', 'diidratado', 'dihidratada', 'dihidratado', 'dipropionato', 'dinitrato',
                            'dissodica', 'dissodico', 'divalproato', 'do', 'dos',
                        'e', 'em', 'enantato', 'esteril', 'estolato',
                        'forma', 'fosfato', 'fumarato',
                        'g',
                        'h', 'hemi', 'hemieptaidratada', 'hemieptaidratado', 'hemifumarato', 'hemiidratado', 'hemipentaidratado', 'hemitartarato', 'heptaidratado',
                            'hexaidratado', 'hidratada', 'hidratado', 'hidroxido',
                        'lactato', 'longa',
                        'magnesica', 'magnesico', 'maleato', 'membrana', 'mesilato', 'micronizada', 'micronizado', 'monofosfato', 'monohidratada',
                            'monoidratada', 'monoidratado', 'mononitrato', 'mucato',
                        'n',
                        'o', 'oxalato', 'oxido',
                        'p', 'palmitato', 'para', 'pentahidratado', 'pentaidratada', 'pentaidratado', 'pivoxila', 'potassica',
                        's', 'sem', 'sesquiidratado', 'sodica', 'sodico', 'succinato', 'sulfato',
                        'tartarato', 'tetraidratado', 'tipo', 'tri', 'tribasico', 'triidratada', 'triidratado', 'trihidratada', 'trihidratado',
                        'v', 'valerato', 'valproato',
                        'zincica']

        tokens = [tok for tok in tokens if tok not in STOPWORDS_AI]

    # Remove words that hinder the identification of presentations
    if rem_stopwords_pr:
        STOPWORDS_PR = ['embalagem', 'agua', 'de', 'para', 'sodio', 'e']
        
        tokens = [tok for tok in tokens if tok not in STOPWORDS_PR]

    # Abbreviate presentation components based on the ANVISA vocabulary
    if abbreviate_prs:
        ANVISA_ABBREVIATOR = {'adaptador': 'adapt',
                        'adesivo': 'ades',
                        'aerossol': 'aer',
                        'agulha': 'agu',
                        'aluminio': 'al',
                        'ambar': 'amb',
                        'ampola': 'amp',
                        'anel': 'anel',
                        'aplicador': 'aplic',
                        'aplicadora': 'aplic',
                        'ativador': 'ativ',
                        'barra': 'bar',
                        'bastao': 'bast',
                        'bisnaga': 'bg',
                        'blister': 'bl',
                        'bolsa': 'bols',
                        'bombeador': 'bomb',
                        'bombona': 'bombo',
                        'bucal': 'buc',
                        'camara': 'cam',
                        'caneta': 'can',
                        'capsula': 'cap',
                        'capilar': 'capi',
                        'carpule': 'car',
                        'conta': 'cgt',
                        'cilindro': 'cil',
                        'colher': 'col',
                        'colutorio': 'colut',
                        'comprimido': 'com',
                        'copo': 'cop',
                        'creme': 'crem',
                        'cartucho': 'ct',
                        'caixa': 'cx',
                        'dermatologica': 'derm',
                        'dermatologico': 'derm',
                        'diluente': 'dil',
                        'diluicao': 'dil',
                        'uterino': 'diu',
                        'dosadora': 'dos',
                        'dura': 'dura',
                        'efervescente': 'efev',
                        'elixir': 'elx',
                        'emplasto': 'empl',
                        'envelope': 'env',
                        'epidural': 'epi',
                        'esmalte': 'esm',
                        'espatula': 'esp',
                        'espuma': 'esp',
                        'espacador': 'espac',
                        'estojo': 'est',
                        'frasco-ampola': 'fa',
                        'fechado': 'fech',
                        'filme': 'fil',
                        'flaconete': 'flac',
                        'frasco': 'fr',
                        'gas': 'gas',
                        'gel': 'gel',
                        'globulo': 'glob',
                        'gomosa': 'gom',
                        'goma': 'goma',
                        'gotas': 'got',
                        'gotejador': 'got',
                        'granulado': 'gran',
                        'articular': 'ia',
                        'arterial': 'iar',
                        'intradermica': 'id',
                        'intramuscular': 'im',
                        'implante': 'impl',
                        'inalacao': 'inal',
                        'inalador': 'inal',
                        'inaladora': 'inal',
                        'inalatoria': 'inal',
                        'infusao': 'infus',
                        'injetavel': 'inj',
                        'irrigacao': 'irr',
                        'intratecal': 'it',
                        'intrauterina': 'iu',
                        'intravenosa': 'iv',
                        'lamina': 'lam',
                        'lenco': 'len',
                        'liberacao': 'lib',
                        'liofilo': 'liof',
                        'liofilizado': 'liof',
                        'liquido': 'liq',
                        'mastigavel': 'mast',
                        'metal': 'met',
                        'emulsao': 'meu',
                        'modificada': 'mod',
                        'mole': 'mole',
                        'nasal': 'nas',
                        'oftalmica': 'oft',
                        'oleo': 'ole',
                        'opaco': 'opc',
                        'oral': 'or',
                        'orodispersivel': 'orodisp',
                        'otologica': 'oto',
                        'ovulo': 'ovl',
                        'papel': 'pap',
                        'pastinha': 'pas',
                        'pasta': 'past',
                        'pincel': 'pinc',
                        'plastico': 'plas',
                        'po': 'po',
                        'pomada': 'pom',
                        'preenchida': 'preenc',
                        'preenchido': 'preenc',
                        'prolongada': 'prol',
                        'pote': 'pt',
                        'rasura': 'ras',
                        'retal': 'ret',
                        'retardada': 'retard',
                        'revestido': 'rev',
                        'sabonete': 'sab',
                        'subcutanea': 'sc',
                        'seringa': 'ser',
                        'sistema': 'sist',
                        'solucao': 'sol',
                        'spray': 'spr',
                        'strip': 'str',
                        'sublingual': 'subl',
                        'supositorio': 'sup',
                        'suspensao': 'sus',
                        'suspencao': 'sus',
                        'tablete': 'table',
                        'tubo': 'tb',
                        'termica': 'term',
                        'transparente': 'trans',
                        'transdermica': 'transd',
                        'transferencia': 'transf',
                        'translucido': 'transl',
                        'uretral': 'uret',
                        'vaginal': 'vag',
                        'valcula': 'valv',
                        'vidro': 'vd',
                        'xampu': 'xamp',
                        'xarope': 'xpe'}
        
        tokens = [ANVISA_ABBREVIATOR.get(tok, tok) for tok in tokens]

    # Removal of repeated words
    if rem_rep_tokens:
        seen = set()

        filtered_tokens = []
        for x in tokens:
            if x in seen:
                continue

            seen.add(x)
            filtered_tokens.append(x)

        tokens = filtered_tokens

    return " ".join(tokens)

def group_cmed(df_cmed, ai_column, verbose = False):
    """
    Creates a dictionary-like DataFrame where the "keys" are pharmaceutical
    active ingredients and the "values" are the indices of CMED rows that
    contain each ingredient. The resulting DataFrame also includes a third
    column with the keys sorted alphabetically.

    Parameters
    ----------
    df_cmed : DataFrame
        DataFrame containing the CMED data.

    ai_column : str
        Name of the column that contains pharmaceutical active ingredient
        information.

    verbose : bool, default False
        Whether to display progress information during processing.

    Returns
    -------
    DataFrame
        A DataFrame where each row represents a pharmaceutical ingredient, its
        associated row indices from the original CMED data, and the ingredient
        name sorted alphabetically.
    """
    # Create a list with all the distinct pharmaceutical active ingredients
    ais = df_cmed[ai_column].unique()

    if verbose:
        ais = tqdm(ais, desc = "Grouping active ingredients")
    
    # Creation of the DataFrame
    keys = []
    keys_sorted = []
    indexes = []

    for key in ais:
        # Find the rows of CMED that have the pharmaceutical ingredient of the
        # current key
        indexes_found = df_cmed.index[df_cmed[ai_column] == key].values

        keys.append(key)
        keys_sorted.append(sort_alphabetically(key))
        indexes.append(np.unique(indexes_found))
    
    df_grouped_cmed = pd.DataFrame({
        'key': keys,
        'key_sorted': keys_sorted,
        'indexes': indexes
    })

    ### Dealing with duplicated key_sorted lines
    duplicated = df_grouped_cmed[df_grouped_cmed.duplicated(subset = ['key_sorted'],
                                                            keep = False)]

    if not duplicated.empty:
        # Aggregate indexes for each duplicated key_sorted
        agg = duplicated.groupby('key_sorted').agg({
            'key': 'first',
            'indexes': lambda idxs: np.unique(np.concatenate(idxs.tolist()))
        }).reset_index()
        
        # Remove old duplicates and add aggregated ones
        df_grouped_cmed = df_grouped_cmed.drop(duplicated.index)
        df_grouped_cmed = pd.concat([df_grouped_cmed, agg], ignore_index =True)

    return df_grouped_cmed.sort_values(by = ['key']).reset_index(drop = True)

def sort_alphabetically(text):
    """
    Returns a string with its words sorted in alphabetical order.

    Parameters
    ----------
    text : str
        Input string whose words will be sorted alphabetically.

    Returns
    -------
    str
        A string with the same words, reordered alphabetically.
    """
    
    return " ".join(sorted(word_tokenize(text)))

def load_notice(path, sep = ';', decimal = ','):
    """
    Load the Public Notice data from a CSV file.

    Parameters
    ----------
    path : str, path object, or file-like object
        Path to a CSV file containing the Public Notice data. Can be a string,
        a PathLike object, or a file-like object with a ``read()`` method.

    sep : str, default ';'
        Character used to separate fields in the CSV file.

    decimal : str, default ','
        Character used as the decimal point in numeric values.

    Returns
    -------
    DataFrame
        A DataFrame containing the data from the Public Notice file.
    """

    return pd.read_csv(path, sep = sep, decimal = decimal)