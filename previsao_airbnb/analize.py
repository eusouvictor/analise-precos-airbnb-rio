import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Caminho absoluto para o arquivo
file_path = os.path.join(os.getcwd(), 'previsao_airbnb', 'listings.csv.gz')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no local correto.")

df = pd.read_csv(file_path)

# Remove símbolos de moeda e vírgulas, e converte para numérico
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)

# Converte a coluna 'price' para numérico (float)
df['price'] = pd.to_numeric(df['price'])

# Remove valores nulos e negativos
df = df.dropna(subset=['price'])
df = df[df['price'] > 0]

# Vamos manter apenas preços menores que R$ 5.000 para não confundir o modelo
df = df[df['price'] < 5000]

# Estas são as colunas que o modelo vai usar para aprender
colunas = ['price', 'neighbourhood_cleansed', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds']

# Criamos um novo DataFrame só com o que importa
df_limpo = df[colunas].copy()

# Vamos ver quantos buracos temos
print("\n----- Valores Vazios antes de limpar -----")
print(df_limpo.isnull().sum())

# Se não diz quantos banheiros/quartos/camas, assumimos que é 1
df_limpo['bathrooms'] = df_limpo['bathrooms'].fillna(1)
df_limpo['bedrooms'] = df_limpo['bedrooms'].fillna(1)
df_limpo['beds'] = df_limpo['beds'].fillna(1)

# Vamos manter apenas os TOP 20 bairros mais comuns
top_20_bairros = df_limpo['neighbourhood_cleansed'].value_counts().nlargest(20).index
df_limpo['neighbourhood_cleansed'] = df_limpo['neighbourhood_cleansed'].apply(lambda x: x if x in top_20_bairros else 'Outros')

# Agora aplicamos o One-Hot Encoding (transforma texto em colunas de 0 e 1)
df_final = pd.get_dummies(df_limpo, columns=['neighbourhood_cleansed', 'room_type'])

# Converter True/False para 1/0 (algumas versões do pandas precisam disso)
df_final = df_final.astype(float)

print("\n----- Visualizando os Dados Prontos para a IA -----")
print(df_final.head())
print("\n----- Novas Colunas Criadas -----")
print(df_final.columns)


# X = Perguntas (Features): Tudo MENOS o preço
X = df_final.drop('price', axis=1)
# y = Resposta (Target): SÓ o preço
y = df_final['price']

# Dividimos 80% para estudar e 20% para a prova
# random_state=42 garante que o sorteio seja sempre o mesmo (para reproduzibilidade)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n----- Tamanho dos Conjuntos -----")
print(f"Treino: {X_train.shape[0]} imóveis")
print(f"Teste:  {X_test.shape[0]} imóveis")

# --- 6. TREINAMENTO (O MODELO ESTUDA) ---
print("\nIniciando treinamento... (pode demorar alguns segundos)")
modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo.fit(X_train, y_train)
print("Treinamento concluído!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ... (seu código de treino continua igual) ...

# --- 7. AVALIAÇÃO PROFISSIONAL ---
previsoes = modelo.predict(X_test)

# ... (seu código de treino continua igual lá pra cima) ...

# --- 7. AVALIAÇÃO TÉCNICA DO MODELO ---
previsoes = modelo.predict(X_test)

# Métricas Padrão
mae = mean_absolute_error(y_test, previsoes)
rmse = np.sqrt(mean_squared_error(y_test, previsoes))
r2 = r2_score(y_test, previsoes)

# --- CÁLCULO DA "ACURÁCIA" (Margem de Aceitação) ---
# Vamos considerar "acerto" se o modelo errar por menos de 20% do valor real
margem_aceitavel = 0.20 
erros_percentuais = np.abs((y_test - previsoes) / y_test)
acertos = np.sum(erros_percentuais <= margem_aceitavel)
acuracia_custom = (acertos / len(y_test)) * 100

print("\n" + "#" * 60)
print("RELATORIO DE DESEMPENHO - MODELO RANDOM FOREST")
print("#" * 60)
print(f"Erro Medio Absoluto (MAE):         R$ {mae:.2f}")
print(f"Raiz do Erro Quadratico (RMSE):    R$ {rmse:.2f}")
print(f"Coeficiente de Determinacao (R2):  {r2:.2%}")
print("-" * 60)
print(f"ACURACIA (Margem de erro <= 20%):  {acuracia_custom:.2f}%")
print("-" * 60)
print("Interpretacao:")
print(f" - O modelo explica {r2:.1%} da variancia dos dados.")
print(f" - Em {acuracia_custom:.1f}% dos casos, a previsao ficou muito proxima do valor real.")
print("#" * 60)

# --- 8. AMOSTRA DE RESULTADOS ---
df_resultado = pd.DataFrame()
df_resultado['Real'] = y_test.values
df_resultado['Previsto'] = previsoes
df_resultado['Dif (R$)'] = df_resultado['Real'] - df_resultado['Previsto']
df_resultado['Erro (%)'] = np.abs((df_resultado['Dif (R$)'] / df_resultado['Real']) * 100)

print("\nAMOSTRA DE 10 PREVISOES ALEATORIAS:")
print("-" * 60)
# Formatação limpa para a tabela não ficar torta
print(df_resultado.sample(10).round(2).to_string(index=False)) 
print("-" * 60 + "\n")