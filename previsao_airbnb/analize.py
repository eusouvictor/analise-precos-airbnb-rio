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

# --- VERIFICAÇÃO FINAL ---
print("\n----- Tudo pronto? (Zeros aqui significam sucesso) -----")
print(df_limpo.isnull().sum())

print("\n----- Tamanho Final do Dataset -----")
print(df_limpo.shape)