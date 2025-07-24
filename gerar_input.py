# -*- coding: utf-8 -*-
"""
======================================================================
SCRIPT AUXILIAR (v1.4) - GERADOR DE ARQUIVO DE SKUs
======================================================================
(...)
Correção v1.4:
- Adicionada a limpeza de espaços em branco (strip) na coluna 'sku'
  logo após a leitura para garantir consistência dos dados.
"""

import pandas as pd
import sys
import random

# --- CONFIGURAÇÃO ---
arquivo_vendas = 'vendas.csv'
arquivo_saida_skus = 'skus_para_previsao.txt'
percentual_amostra = 0.5

# --- EXECUÇÃO DO SCRIPT ---
print("--- Iniciando o gerador de arquivo de SKUs ---")
try:
    print(f"Lendo o arquivo '{arquivo_vendas}'...")
    df = pd.read_csv(arquivo_vendas, usecols=['sku'], dtype={'sku': str})

    # ==================================================================
    # CORREÇÃO CRÍTICA APLICADA AQUI
    # Remove espaços em branco do início e do fim de cada SKU.
    # ==================================================================
    df['sku'] = df['sku'].str.strip()

    lista_todos_os_skus = list(df['sku'].unique())
    total_skus = len(lista_todos_os_skus)
    print(f"Total de {total_skus} SKUs únicos e limpos encontrados.")

    tamanho_amostra = int(total_skus * percentual_amostra)
    print(f"Selecionando uma amostra aleatória de {percentual_amostra:.0%} ({tamanho_amostra} SKUs)...")
    skus_amostra = random.sample(lista_todos_os_skus, k=tamanho_amostra)
    
    with open(arquivo_saida_skus, 'w') as f:
        for sku in skus_amostra:
            f.write(f"{sku}\n")

    print("\n" + "="*70)
    print("ARQUIVO DE SKUs GERADO COM SUCESSO!")
    print(f"O arquivo '{arquivo_saida_skus}' foi criado com {tamanho_amostra} SKUs.")
    print("Use este nome de arquivo no prompt do script principal.")
    print("="*70)

except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_vendas}' não foi encontrado.")
    sys.exit()
except Exception as e:
    print(f"\nOcorreu um erro inesperado: {e}")