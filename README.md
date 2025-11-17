# 🤖 Análise e Previsão de Preços do Airbnb no Rio de Janeiro

Este repositório contém o código e a análise do projeto de Machine Learning para prever preços de aluguel no Airbnb na cidade do Rio de Janeiro.

## 🎯 Objetivo do Projeto

O objetivo principal é duplo:
1.  Desenvolver um modelo de Machine Learning (Regressão) capaz de prever o preço de um imóvel com base em suas características (localização, tipo de quarto, comodidades).
2.  [cite_start]Analisar criticamente o modelo e os dados, investigando potenciais riscos éticos, como viés algorítmico e discriminação, conforme discutido em nosso artigo de pesquisa. [cite: 1, 9]

## ⚠️ Considerações Éticas

Este projeto não ignora os riscos do ML. Nosso modelo é um estudo de caso para entender como os dados podem perpetuar desigualdades. [cite_start]Nossa pesquisa  identificou que:

* [cite_start]**Viés Algorítmico:** Modelos treinados com dados históricos podem aprender e amplificar vieses existentes, subvalorizando imóveis em bairros de maioria não-branca. [cite: 27, 37]
* [cite_start]**Privacidade:** O uso de datasets públicos como o Inside Airbnb levanta questões sobre a privacidade de anfitriões e usuários. [cite: 21, 22]

## 🛠️ Metodologia

1.  **Fonte de Dados:** [Inside Airbnb](http://insideairbnb.com/get-the-data/) - Dataset `listings.csv` do Rio de Janeiro (data da coleta).
2.  **Limpeza e Tratamento:** (Descrever os passos de limpeza: remoção de outliers, tratamento de valores nulos, etc.)
3.  **Engenharia de Features:** (Descrever as features usadas: `neighbourhood`, `room_type`, `accommodates`, etc.)
4.  **Modelagem:** (Listar os algoritmos usados, ex: Random Forest, XGBoost)
5.  **Avaliação:** (Métrica usada, ex: Mean Absolute Error - MAE)

## 🚀 Resultados e Como Usar

(Aqui você colocará a conclusão)

Para reproduzir este estudo, siga os passos:

1.  Clone o repositório:
    ```bash
    git clone [URL_DO_SEU_REPO]
    ```
2.  Instale as dependências:
    ```bash
    pip install pandas scikit-learn jupyterlab
    ```
3.  Abra e execute o notebook:
    ```bash
    jupyter-lab "Seu_Notebook.ipynb"
    ```

## 👥 Autores

* [cite_start]João Lucas de Lima Correia [cite: 3]
* [cite_start]Pedro Henrique Geroncio Bispo Dourado [cite: 4]
* [cite_start]Huan Victor França Do Noscimento [cite: 5]
* [cite_start]Mateus Cintra De Brito [cite: 5]
