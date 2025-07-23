# -*- coding: utf-8 -*-
"""
Script de Previsão de Demanda para Gerenciamento de Estoque.

Este script realiza as seguintes etapas:
1.  Gera um arquivo de dados de vendas fictício ('vendas.csv').
2.  Carrega e pré-processa os dados.
3.  Divide os dados em conjuntos de treino e teste.
4.  Treina e compara dois modelos de regressão: Regressão Linear e Random Forest.
5.  Implementa técnicas de interpretabilidade para o melhor modelo.
6.  Avalia o desempenho dos modelos usando o Erro Absoluto Médio (MAE).
7.  Usa o melhor modelo para prever a demanda para os próximos 7 dias.
8.  (Bônus) Visualiza a demanda real vs. prevista.
9.  (Bônus) Permite a previsão para um SKU específico fornecido pelo usuário.
"""

# --- ETAPA 0: Importação de Bibliotecas e Geração de Dados Fictícios ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import os

print("--- Iniciando o Projeto de Previsão de Demanda ---")

def gerar_dados_ficticios(caminho_arquivo='vendas.csv'):
    """Gera um arquivo CSV com dados de vendas fictícios se ele não existir."""
    if os.path.exists(caminho_arquivo):
        print(f"Arquivo '{caminho_arquivo}' já existe. Usando o arquivo existente.")
        return

    print(f"Gerando dados fictícios em '{caminho_arquivo}'...")
    # Criando um range de datas de 2 anos
    datas = pd.to_datetime(pd.date_range(start='2023-01-01', end='2024-12-31'))
    skus = ['SKU_A', 'SKU_B', 'SKU_C']
    dados = []

    for sku in skus:
        for data in datas:
            # Padrão semanal (vendas maiores no fim de semana)
            fator_semanal = (data.dayofweek + 4) % 7
            # Tendência de crescimento ao longo do tempo
            fator_tendencia = 1 + (data - datas[0]).days / 1000
            # Vendas base com alguma aleatoriedade
            vendas_base = np.random.randint(10, 30)
            vendas = int(vendas_base * fator_tendencia + fator_semanal * 3 + np.random.randn() * 3)
            dados.append({'sku': sku, 'data': data, 'vendas': max(0, vendas)})

    df = pd.DataFrame(dados)
    df.to_csv(caminho_arquivo, index=False)
    print("Dados fictícios gerados com sucesso.")

# Gera os dados antes de começar
gerar_dados_ficticios('vendas.csv')


# --- ETAPA 1: Carregamento dos Dados ---
print("\n--- ETAPA 1: Carregando os Dados ---")
try:
    dados = pd.read_csv('vendas.csv', parse_dates=['data'])
    print("Dados carregados com sucesso.")
    print("Primeiras 5 linhas dos dados:")
    print(dados.head())
    print("\nInformações do DataFrame:")
    dados.info()
except FileNotFoundError:
    print("ERRO: Arquivo 'vendas.csv' não encontrado. Execute a geração de dados primeiro.")
    exit()


# --- ETAPA 2: Pré-processamento e Engenharia de Atributos ---
print("\n--- ETAPA 2: Pré-processamento e Engenharia de Atributos ---")

def preprocessar_dados(df):
    """Cria novos atributos a partir da data para o modelo."""
    df_proc = df.copy()
    df_proc['dia_da_semana'] = df_proc['data'].dt.dayofweek
    df_proc['dia_do_mes'] = df_proc['data'].dt.day
    df_proc['mes'] = df_proc['data'].dt.month
    df_proc['ano'] = df_proc['data'].dt.year
    df_proc['semana_do_ano'] = df_proc['data'].dt.isocalendar().week.astype(int)
    
    # Lag feature: vendas da semana anterior (importante para capturar sazonalidade semanal)
    df_proc['vendas_lag_7'] = df_proc.groupby('sku')['vendas'].shift(7)
    
    # Lidando com valores ausentes criados pelo lag
    df_proc = df_proc.dropna()

    # One-Hot Encoding para a variável categórica 'sku'
    df_proc = pd.get_dummies(df_proc, columns=['sku'], drop_first=True)
    
    return df_proc

dados_proc = preprocessar_dados(dados)
print("Dados pré-processados. Novos atributos criados.")
print("Exemplo de dados processados:")
print(dados_proc.head())


# --- ETAPA 3: Divisão dos Dados em Treinamento e Teste ---
print("\n--- ETAPA 3: Divisão dos Dados ---")
# Para séries temporais, a divisão não deve ser aleatória.
# Vamos usar os últimos 3 meses de dados como conjunto de teste.
dados_proc = dados_proc.set_index('data')
data_corte = dados_proc.index.max() - pd.DateOffset(months=3)

treino = dados_proc[dados_proc.index <= data_corte]
teste = dados_proc[dados_proc.index > data_corte]

X_treino = treino.drop('vendas', axis=1)
y_treino = treino['vendas']
X_teste = teste.drop('vendas', axis=1)
y_teste = teste['vendas']

print(f"Dados divididos em treino e teste.")
print(f"Tamanho do conjunto de treino: {len(X_treino)} amostras")
print(f"Tamanho do conjunto de teste: {len(X_teste)} amostras")
print(f"Data de corte para divisão: {data_corte.date()}")


# --- ETAPA 4: Treinamento e Comparação de Modelos ---
print("\n--- ETAPA 4: Treinamento e Comparação de Modelos ---")
modelos = {
    "Regressão Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

for nome, modelo in modelos.items():
    print(f"Treinando modelo: {nome}...")
    modelo.fit(X_treino, y_treino)
    print(f"{nome} treinado.")


# --- ETAPA 5 e 6: Avaliação de Desempenho e Interpretação ---
print("\n--- ETAPA 5 & 6: Avaliação e Interpretação dos Modelos ---")
resultados = {}
for nome, modelo in modelos.items():
    previsoes = modelo.predict(X_teste)
    mae = mean_absolute_error(y_teste, previsoes)
    resultados[nome] = mae
    print(f"Desempenho do {nome}:")
    print(f"  - Erro Absoluto Médio (MAE): {mae:.2f} unidades")

# Identificando o melhor modelo
melhor_modelo_nome = min(resultados, key=resultados.get)
melhor_modelo = modelos[melhor_modelo_nome]
print(f"\nO melhor modelo foi '{melhor_modelo_nome}' com o menor MAE.")

# Interpretação do Melhor Modelo
print(f"\n--- Interpretação do Melhor Modelo ({melhor_modelo_nome}) ---")
if melhor_modelo_nome == "Random Forest":
    importancias = pd.Series(melhor_modelo.feature_importances_, index=X_treino.columns)
    print("Importância dos Atributos (Feature Importances):")
    print(importancias.sort_values(ascending=False))
    
    # Plot de importância
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importancias.sort_values(ascending=False), y=importancias.sort_values(ascending=False).index)
    plt.title(f'Importância dos Atributos - {melhor_modelo_nome}')
    plt.xlabel('Importância')
    plt.ylabel('Atributo')
    plt.tight_layout()
    plt.show()

elif melhor_modelo_nome == "Regressão Linear":
    coeficientes = pd.Series(melhor_modelo.coef_, index=X_treino.columns)
    print("Coeficientes do Modelo (Impacto de cada atributo):")
    print(coeficientes)


# --- ETAPA 7: Previsão para os Próximos 7 Dias ---
print("\n--- ETAPA 7: Previsão para os Próximos 7 Dias ---")

def prever_futuro(modelo, dados_originais, dias_a_prever=7):
    """Gera um DataFrame com as previsões de demanda para os próximos dias."""
    ultima_data = dados_originais['data'].max()
    datas_futuras = pd.to_datetime(pd.date_range(start=ultima_data + pd.Timedelta(days=1), periods=dias_a_prever))
    
    previsoes_futuras = []
    skus_unicos = dados_originais['sku'].unique()

    for sku in skus_unicos:
        # Pega os dados históricos apenas deste SKU para obter o lag correto
        dados_sku_historico = dados_originais[dados_originais['sku'] == sku].copy()
        
        df_futuro = pd.DataFrame({'data': datas_futuras})
        df_futuro['sku'] = sku
        
        # Recria as features de lag usando os dados históricos mais recentes
        # Concatena para poder calcular o lag de forma contínua
        df_combinado = pd.concat([dados_sku_historico, df_futuro], ignore_index=True)
        df_combinado['vendas_lag_7'] = df_combinado['vendas'].shift(7)
        
        # Filtra para manter apenas as datas futuras, que agora têm o lag calculado
        df_futuro_com_lag = df_combinado[df_combinado['data'].isin(datas_futuras)].copy()
        
        # Aplica o mesmo pré-processamento
        df_futuro_proc = preprocessar_dados(df_futuro_com_lag.drop(columns=['vendas']))

        # Garante que as colunas do dataframe futuro correspondem às colunas de treino
        colunas_modelo = modelo.feature_names_in_
        for col in colunas_modelo:
            if col not in df_futuro_proc.columns:
                df_futuro_proc[col] = 0 # Adiciona colunas de SKU faltantes
        df_futuro_proc = df_futuro_proc[colunas_modelo] # Reordena para garantir consistência

        previsao = modelo.predict(df_futuro_proc.drop('data', axis=1))
        
        df_previsao = pd.DataFrame({
            'data': datas_futuras,
            'sku': sku,
            'previsao_vendas': np.round(previsao).astype(int)
        })
        previsoes_futuras.append(df_previsao)
        
    return pd.concat(previsoes_futuras)

previsoes_7_dias = prever_futuro(melhor_modelo, dados)
print("Previsão de demanda para os próximos 7 dias:")
print(previsoes_7_dias)


# --- BÔNUS 1: Visualização da Demanda Real x Prevista ---
print("\n--- BÔNUS 1: Visualização Real vs. Prevista ---")

# Criando um DataFrame para visualização
df_viz = y_teste.to_frame(name='vendas_reais')
df_viz['previsoes'] = melhor_modelo.predict(X_teste)
df_viz = df_viz.join(dados_proc[[col for col in dados_proc.columns if 'sku_' in col]])

# Mapeando de volta para o nome do SKU
sku_cols = [col for col in dados_proc.columns if 'sku_' in col]
df_viz['sku'] = 'SKU_A' # SKU base (que não tem coluna dummy)
for col in sku_cols:
    sku_name = col.replace('sku_', '')
    df_viz.loc[df_viz[col] == 1, 'sku'] = sku_name

# Plotando para um SKU específico para clareza
sku_para_plotar = 'SKU_B'
df_plot = df_viz[df_viz['sku'] == sku_para_plotar]

plt.figure(figsize=(15, 7))
plt.plot(df_plot.index, df_plot['vendas_reais'], label='Vendas Reais', marker='.', linestyle='-')
plt.plot(df_plot.index, df_plot['previsoes'], label=f'Previsões ({melhor_modelo_nome})', marker='x', linestyle='--')
plt.title(f'Comparação: Vendas Reais vs. Previstas para {sku_para_plotar}')
plt.xlabel('Data')
plt.ylabel('Número de Vendas')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- BÔNUS 2: Função para Previsão por SKU Específico ---
print("\n--- BÔNUS 2: Função de Previsão por SKU ---")

def prever_demanda_por_sku(sku_id, modelo, dados_originais):
    """Função que permite ao usuário especificar um ID de produto para previsão."""
    if sku_id not in dados_originais['sku'].unique():
        return f"Erro: SKU '{sku_id}' não encontrado nos dados históricos."
    
    print(f"Gerando previsão de 7 dias para o produto: {sku_id}")
    previsoes_completas = prever_futuro(modelo, dados_originais)
    previsao_especifica = previsoes_completas[previsoes_completas['sku'] == sku_id]
    
    return previsao_especifica

# Exemplo de uso da função com interação do usuário
try:
    sku_usuario = input("Digite o ID do produto (ex: SKU_A, SKU_B, SKU_C) para ver a previsão de 7 dias: ").strip()
    previsao_usuario = prever_demanda_por_sku(sku_usuario, melhor_modelo, dados)
    print("\nResultado da Previsão Específica:")
    print(previsao_usuario)
except Exception as e:
    print(f"Ocorreu um erro: {e}")

print("\n--- Projeto concluído com sucesso! ---")