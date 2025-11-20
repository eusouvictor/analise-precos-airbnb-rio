import os
import pandas as pd

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

# Exibe informações sobre a coluna 'price'
print("\n----- Preço limpo (Primeiras 5 linhas) -----")
print(df['price'].head())

# Verifica o tipo de dado da coluna 'price'
print("\n----- Tipo de dado da coluna (Deve ser Float64) -----")
print(df['price'].dtype)

# Estatísticas descritivas da coluna 'price'
print("\n----- Estatísticas descritivas do preço -----")
print(df['price'].describe())