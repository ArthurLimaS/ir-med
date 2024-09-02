import etl_functions as etl
# import math
import numpy as np
from jaro import jaro_winkler_metric
from nltk.tokenize import word_tokenize
from tqdm import tqdm



# Retornar as listas de tokens relevantes para classificar principios ativos (tokens_pr) e apresentações (tokens_apr)
def get_token_list(df_cmed):
    tokens_pr = ""
    tokens_apr = ""
    for _, row in tqdm(df_cmed.iterrows()):
        tokens_pr += row['principio_ativo'] + " "
        tokens_apr += row['apresentacao']

    tokens_pr = np.unique(word_tokenize(tokens_pr))
    tokens_apr = np.unique(word_tokenize(tokens_apr))
    
    return tokens_pr, tokens_apr



# Quebra o campo 'descrição' de um edital 
# principio_ativo/produto e de apresentacao
def sep_descricao(descricao, tokens_pr, tokens_apr):
    pr = ""
    apr = ""

    for tok in word_tokenize(descricao):
        if tok in tokens_pr:
            pr += tok + " "
        if tok in tokens_apr:
            apr += tok + " "
        
    return pr, apr



# Macro função que realiza o processo inteiro de classificação
def coletor_eans(df_cmed, agrupamento_pr, pr, apr, und):

    # Classificação de principio ativo
    grupo, _ = match_pr(agrupamento_pr, pr)

    # Coleta dos medicamentos da CMED que possuem o principio ativo apontado
    df_cmed_group = df_cmed.iloc[agrupamento_pr[agrupamento_pr['key'] == grupo].reset_index()['indexes'][0]]
    
    return coletor_eans_apr(df_cmed_group, pr, apr, und, grupo)



# Função de classificação do principio ativo
def match_pr(agrupamento_pr, pr):
    pr = etl.ordem_alfabetica(pr)
    best_match = -1
    best_match_key = ""

    for i in range(len(agrupamento_pr['key_sorted'])):
        key_sorted = agrupamento_pr['key_sorted'][i]

        # Cálculo da próximidade
        metric = jaro_winkler_metric(pr, key_sorted)

        if metric >= best_match:
            best_match = metric
            best_match_key = agrupamento_pr['key'][i]

    dados_extras = {'string_pr': pr,
                    'best_match': best_match}

    return (best_match_key, dados_extras)



# Função de classificação da apresentação
def coletor_eans_apr(df_cmed_group, pr, apr, und, grupo):

    ### Filtrar utilizando as palavras que estão presentes em apr e und
    conjs = get_sets_apr(apr)
    und_conjs = get_sets_apr(und)
    conjs.extend(und_conjs)

    best_count = 0
    best_matchs = []

    for idx_cmed, row_cmed in df_cmed_group.iterrows():
        # Checar se a apresentação tem os valores encontrados no doc da licitação
        count = 0
        tokens_cmed = word_tokenize(row_cmed['apresentacao'])

        for conj in conjs:
            # Verificar se o conjunto possui só um elemento
            if len(conj) == 1:
                # Verificar se esse elemento está presente no texto
                if conj[0] in tokens_cmed:
                    count += 1

            else:
                # Verificar se essa sequência está presente no texto
                if conj[0] in tokens_cmed:
                    check = True
                    initial_index = tokens_cmed.index(conj[0])
                    
                    for i in range(1, len(conj)):
                        index_atual = (i+initial_index)
                        
                        if index_atual >= len(tokens_cmed):
                            check = False
                            break
                        elif conj[i] != tokens_cmed[index_atual]:
                            check = False
                            break
                                
                    if check:
                        count += 1

        if count > best_count:
            best_count = count
            best_matchs = [idx_cmed]
        elif count == best_count:
            best_matchs.append(idx_cmed)


    ### Filtrar utilizando as palavras que tão presentes em unid
    # best_count = 0
    # best_matchs = []


    # for idx_cmed, row_cmed in df_cmed_group.loc[good_matchs].iterrows():
    #     # Checar se a apresentação tem os valores encontrados no doc da licitação
    #     count = 0
    #     tokens_cmed = word_tokenize(row_cmed['apresentacao'])

    #     for conj in apr_conjs:
    #         # Verificar se o conjunto possui só um elemento
    #         if len(conj) == 1:
    #             # Verificar se esse elemento está presente no texto
    #             if conj[0] in tokens_cmed:
    #                 count += 1

    #         else:
    #             # Verificar se essa sequência está presente no texto
    #             if conj[0] in tokens_cmed:
    #                 check = True
    #                 initial_index = tokens_cmed.index(conj[0])
                    
    #                 for i in range(1, len(conj)):
    #                     index_atual = (i+initial_index)
                        
    #                     if index_atual >= len(tokens_cmed):
    #                         check = False
    #                         break
    #                     else:
    #                         if conj[i] != tokens_cmed[index_atual]:
    #                             check = False
    #                             break
                                
    #                 if check:
    #                     count += 1

    #     if count > best_count:
    #         best_count = count
    #         best_matchs = [idx_cmed]
    #     elif count == best_count:
    #         best_matchs.append(idx_cmed)


    return (best_matchs, {'quant_matched': len(best_matchs),
                          'quant_grupo': len(df_cmed_group),
                          'perc_red_conj': (1 - (len(best_matchs) / len(df_cmed_group))),
                          'string_pr': pr,
                          'pr_encontrado': grupo,
                          'string_apr': apr})



def get_sets_apr(apr):
    tokens = word_tokenize(apr)
    n_tokens = len(tokens)
    func = lambda x : (x**2 + x) / 2
    quant_verificacoes = int(func(n_tokens))

    results = []
    group_size = 1
    reduction_value = 0
    for i in range(quant_verificacoes):
        x = i - reduction_value
        
        match = (n_tokens - group_size + 1)
        if x >= match:
            x -= (n_tokens - group_size + 1)
            reduction_value += (n_tokens - group_size + 1)
            group_size += 1

        aux = []
        indexes = []
        for j in range(group_size):
            indexes.append(x+j)
            aux.append(tokens[x+j])

        results.append(aux)

    return results