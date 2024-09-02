import nltk
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from unidecode import unidecode




# Load the CMED dataset from a file
def load_cmed(path, preprocess = False):
    # Load the .csv
    df_cmed = pd.read_csv(path, sep = ";")

    # Adjust columns names
    df_cmed.rename(str.lower, axis = 'columns', inplace = True)
    df_cmed.rename(unidecode, axis = 'columns', inplace = True)
    df_cmed.rename(columns = {'substancia': 'principio_ativo',
                              'ean 1': 'ean_1',
                              'ean 2': 'ean_2',
                              'ean 3': 'ean_3'}, inplace = True)
    
    # Apply the preprocess function to the columns 'principio_ativo' and 'apresentacao'
    if preprocess:
        print("Preprocessing CMED")

        for idx, row in tqdm(df_cmed.iterrows()):
            df_cmed.at[idx, 'principio_ativo'] = preprocessing_function(row['principio_ativo'], rem_nums = True, rem_stopwords_ai = True,
                                                                        correct_ai = True, rem_rep_tokens = True)
            
            df_cmed.at[idx, 'apresentacao'] = preprocessing_function(row['apresentacao'], rem_stopwords_pr = True)

    return df_cmed



def preprocessing_function(text, correct_ai = False, rem_nums = False, rem_stopwords_ai = False,
                           rem_stopwords_pr = False, abbreviate_prs = True, rem_rep_tokens = False):
    text = text.lower()                   # Apply lowercase
    text = unidecode(text)                # Remove acentuacion
    text = re.sub('\W',' ', text)         # Removes specials characters and leaves only words
    text = re.sub(r'http\S+', '', text)   # Removes URLs with http
    text = re.sub(r'www\S+', '', text)    # Removes URLs with www

    tokens = word_tokenize(text)

    # Correct incorrect writing of pharmaceutical ingredients
    if correct_ai:
        correcoes = {'acilovir': 'aciclovir',
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
        
        triggers = correcoes.keys()
        tokens = [correcoes[tok] if tok in triggers else tok for tok in tokens]

    # Insert blank space between numbers and words
    text = ""
    for tok in tokens:
        match = re.split(r'(\d+)', tok)
        if (len(match) > 1):
            for m in match:
                text += m + " "
        else:
            text += tok + " "

    # Remove numbers
    if rem_nums:
        text = re.sub('\d', ' ', text)

    tokens = word_tokenize(text)

    # Remove words that hinder the identification of pharmaceutical ingredients
    if rem_stopwords_ai:
        stopwords_ai = ['a', 'acetato', 'acido', 'anidra',
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

        tokens = [tok for tok in tokens if tok not in stopwords_ai]

    # Remove words that hinder the identification of presentations
    if rem_stopwords_pr:
        stopwords_pr = ['embalagem', 'agua', 'de', 'para', 'sodio', 'e']
        tokens = [tok for tok in tokens if tok not in stopwords_pr]

    # Abreviate presentation components based on the ANVISA vocabulary
    if abbreviate_prs:
        abreviator = {'adaptador': 'adapt',
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
        forms = abreviator.keys()

        tokens = [abreviator[tok] if tok in forms else tok for tok in tokens]

    # Removal of repeared words
    if rem_rep_tokens:
        indexes = np.unique(tokens, return_index=True)[1]
        tokens = [tokens[index] for index in sorted(indexes)]

    text = " ".join(tokens)

    return text


# Creates a dict like DataFrame where the "keys" are the pharmaceutical ingredients
# and the "values" are the indexes of CMED rows that have that ingredient
def grouped_cmed(df_cmed):

    # Create a list with all the distinct pharmaceutical ingredients    
    ais = np.unique(df_cmed['principio_ativo'])
    
    ### Creation of the DataFrame
    keys = []
    keys_sorted = []
    indexes = []

    print("Creation of the grouped-cmed DataFrame")
    for key in tqdm(ais):
        # Find the rows of CMED that have the pharmaceutical ingredient stored in key
        indexes_found = df_cmed.index[df_cmed['principio_ativo'] == key].values

        keys.append(key)
        keys_sorted.append(sort_alphabetically(key))
        indexes.append(np.unique(indexes_found))
    
    data = {'key': keys,
            'key_sorted': keys_sorted,
            'indexes': indexes}

    df_grouped_cmed = pd.DataFrame(data)

    ### Dealing with duplicated lines

    duplicated = df_grouped_cmed[df_grouped_cmed.duplicated(subset=['key_sorted'], keep=False)]

    keys = []
    keys_sorted = []
    indexes = []

    print("Dealing with duplicated pharmaceutical ingredients")
    for ksort in tqdm(np.unique(duplicated['key_sorted'])):
        subset = duplicated[duplicated['key_sorted'] == ksort].reset_index(drop = True)

        keys.append(subset['key'][0])
        keys_sorted.append(ksort)
        
        new_indexes = []
        for ind in subset['indexes']:
            new_indexes.extend(ind)

        indexes.append(np.unique(new_indexes))


    data = {'key': keys,
            'key_sorted': keys_sorted,
            'indexes': indexes}
    new_lines = pd.DataFrame(data).sort_values(by='key_sorted')
    df_grouped_cmed.drop_duplicates(subset=['key_sorted'], keep = False, inplace = True)
    df_grouped_cmed = pd.concat([df_grouped_cmed, new_lines], ignore_index = True)

    return df_grouped_cmed.sort_values(by=['key']).reset_index(drop = True)



# Returns a string with its words in alphabetical order
def sort_alphabetically(text):
    tokens = word_tokenize(text)

    if len(tokens) > 1:
        tokens.sort()
        text = " ".join(tokens)
    
    return text



# Load the .csv with the data extracted from a public notice
def load_notice(path, drop_columns, desc_column, und_column, sep = ';', decimal = ',', preprocess = False):
    
    # Load the .csv
    df_le = pd.read_csv(path, sep = sep, decimal = decimal)

    # Adjust columns names
    df_le.drop(columns = drop_columns, inplace = True)
    df_le.rename(str.lower, axis='columns', inplace = True)
    df_le.rename(unidecode, axis='columns', inplace = True)
    df_le['original_desc'] = df_le[df_le.columns[desc_column]]

    # Apply the preprocess function to the columns 'descrição' and 'unidade'
    if preprocess:
        print("Pré-processamento do edital")
        for idx, row in tqdm(df_le.iterrows()):
            df_le.at[idx, 'original_desc'] = re.sub('\n', '', row['original_desc'])

            df_le.at[idx, df_le.columns[desc_column]] = preprocessing_function(row[desc_column], correct_ai = True, rem_rep_tokens = True)
            
            df_le.at[idx, df_le.columns[und_column]] = preprocessing_function(row[und_column], rem_stopwords_pr = True)
            
    # Creation of the column where the indices of the CMED will be stored
    df_le['cmed_indexes'] = ""

    return df_le











