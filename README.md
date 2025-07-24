# Projeto de Previsão de Demanda Agregada para Varejo

## 1. Visão Geral do Projeto

Este projeto implementa uma solução completa de ponta a ponta para a previsão de demanda em um cenário de varejo. O objetivo principal é fornecer previsões acuradas para auxiliar no gerenciamento de estoque, otimizando custos e evitando rupturas.

O script foi desenvolvido para ser robusto e lidar com um desafio comum no varejo: a **demanda intermitente**, onde muitos produtos não possuem vendas diárias, tornando a previsão por item individual imprecisa e computacionalmente inviável.

### Estratégia Adotada

Para superar o desafio da demanda esparsa, a abordagem central deste projeto é a **agregação de dados**:
1.  **Soma das Vendas:** As vendas de todos os produtos são somadas para cada dia, criando uma única e robusta série temporal de "demanda total".
2.  **Modelagem Agregada:** Modelos de previsão de séries temporais são treinados para prever essa demanda total, que é mais estável e possui padrões mais claros.
3.  **Alocação Proporcional:** Para previsões de grupos específicos de produtos (bônus), a previsão total é alocada com base na participação histórica de vendas do grupo selecionado.

---

## 2. Tecnologias Utilizadas

* **Python 3.11+**
* **Bibliotecas Principais:**
    * `pandas`: Para manipulação e limpeza de dados.
    * `statsmodels` & `pmdarima`: Para modelagem com SARIMA e Auto-SARIMA.
    * `prophet`: Para modelagem com o Prophet do Facebook.
    * `scikit-learn`: Para cálculo da métrica de avaliação (MAE).
    * `matplotlib`: Para visualização de dados e resultados.

---

## 3. Estrutura do Projeto

O projeto é composto pelos seguintes arquivos:

* `previsao_final.py`: O script principal que executa todo o fluxo de análise, desde o carregamento dos dados até a previsão e visualização.
* `gerar_input.py`: Um script auxiliar para gerar um arquivo de texto com uma lista de SKUs, facilitando o teste da funcionalidade de previsão filtrada.
* `vendas.csv`: **(Necessário fornecer)** O arquivo de entrada contendo os dados históricos de vendas.
* `skus_para_previsao.txt`: **(Gerado pelo `gerar_input.py`)** Arquivo de texto contendo a lista de SKUs para a previsão filtrada.
* `requirements.txt`: Arquivo com a lista de dependências do projeto.

---

## 4. Como Executar o Projeto

Siga os passos abaixo para executar a análise completa.

### Passo 1: Pré-requisitos

Certifique-se de ter o Python 3 instalado em seu sistema.

### Passo 2: Instalação das Dependências

1.  Crie um arquivo chamado `requirements.txt` no diretório do projeto com o seguinte conteúdo:
    ```txt
    pandas
    matplotlib
    scikit-learn
    statsmodels
    pmdarima
    prophet
    ```
2.  Instale todas as bibliotecas de uma só vez executando o seguinte comando no seu terminal:
    ```bash
    pip install -r requirements.txt
    ```

### Passo 3: Preparação dos Dados

1.  Coloque seu arquivo de dados de vendas no mesmo diretório do projeto.
2.  Certifique-se de que o arquivo se chama `vendas.csv`.
3.  O arquivo deve conter, no mínimo, as colunas: `sku`, `data` (formato AAAA-MM-DD) e `vendas`.

### Passo 4: Execução

O fluxo foi projetado para ser robusto, usando arquivos para entrada de dados em massa.

**A. Gerar uma lista de SKUs para a previsão filtrada (Opcional, para teste do bônus):**

Execute o script auxiliar para criar um arquivo `skus_para_previsao.txt` com uma amostra de 50% dos seus produtos.

```bash
python gerar_input.py 
```

**B. Executar o script principal de previsão:**

Execute o script principal. Ele realizará todas as etapas de análise e, ao final, solicitará o arquivo de SKUs gerado no passo anterior.

```bash
python previsao_final.py 
```

**C. Interagir com o script:**

1.  O script irá treinar os modelos, exibir os resultados de performance, a interpretação do melhor modelo e a previsão em texto para os próximos 7 dias.
2.  Ele também irá gerar um gráfico consolidado com os resultados.
3.  Ao final, você verá o seguinte prompt no terminal:
```bash
>> BÔNUS: PREVISÃO PARA UM GRUPO ESPECÍFICO DE PRODUTOS
Digite o caminho para o arquivo de SKUs (ex: skus_para_previsao.txt) ou deixe em branco para pular: 
```
4. Digite `skus_para_previsao.txt` e pressione Enter para ver a previsão para o grupo de produtos do arquivo.

## 5. Como o Projeto Atende aos Requisitos

Este projeto foi cuidadosamente construído para satisfazer todos os pontos da tarefa solicitada.

| Requisito | Como foi Atendido |
| :--- | :--- |
| **1. Carregar Dados** | O script carrega o `vendas.csv` usando o `pandas`, convertendo a coluna de data e tratando potenciais problemas de qualidade de dados, como SKUs nulos ou com espaços em branco. |
| **2. Pré-processar Dados** | O pré-processamento é robusto: a estratégia de **agregação de dados** é aplicada para criar a série de demanda total, superando o desafio da demanda intermitente por produto. |
| **3. Dividir Dados** | Os dados são divididos cronologicamente em conjuntos de treino e teste, o que é a prática correta e essencial para modelos de séries temporais, evitando vazamento de dados do futuro. |
| **4. Treinar e Comparar Modelos**| São treinados e comparados **três modelos** especialistas em séries temporais: **Prophet, SARIMA com parâmetros manuais e Auto-SARIMA**, superando o requisito mínimo de dois modelos. |
| **5. Interpretação do Modelo**| Técnicas de interpretação específicas são implementadas: para o **Prophet**, é plotado o gráfico de componentes (tendência e sazonalidade); para o **SARIMA**, é exibido o sumário estatístico completo com coeficientes e testes de diagnóstico. |
| **6. Avaliar Desempenho** | O desempenho dos três modelos é avaliado no conjunto de teste usando o **Erro Absoluto Médio (MAE)**, uma métrica relevante e de fácil interpretação para o negócio (erro em unidades vendidas). O melhor modelo é selecionado automaticamente. |
| **7. Prever Próximos 7 Dias**| O melhor modelo treinado é utilizado para gerar uma previsão clara, em formato de tabela, para a demanda total dos próximos 7 dias. |
| **Bônus 1: Visualização** | Um gráfico consolidado e de alta qualidade é gerado, mostrando os dados históricos, as previsões de todos os modelos no período de teste e a previsão futura do melhor modelo, facilitando a análise visual. |
| **Bônus 2: Previsão por Produto**| A funcionalidade foi implementada de forma robusta e escalável. O usuário pode fornecer um **arquivo de texto com uma lista de SKUs** para receber uma previsão filtrada para aquele grupo específico, resolvendo as limitações de input do terminal. |