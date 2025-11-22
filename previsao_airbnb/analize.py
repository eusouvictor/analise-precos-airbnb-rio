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

# --- 7. AVALIAÇÃO (A PROVA) ---
# O modelo tenta adivinhar os preços dos 20% que ele nunca viu
previsoes = modelo.predict(X_test)

# Calculamos o Erro Médio Absoluto (MAE)
# Ex: Se o real é 300 e ele chutou 350, o erro é 50.
erro_medio = mean_absolute_error(y_test, previsoes)

print("\n----- RESULTADO FINAL -----")
print(f"O erro médio do nosso modelo é de: R$ {erro_medio:.2f}")

# Comparação visual rápida
print("\nExemplo de Previsões vs Realidade:")
comparacao = pd.DataFrame({'Real': y_test, 'Previsto': previsoes}).head(10)
print(comparacao)

# --- 10. O DETECTOR DE PREÇOS ABUSIVOS (PARA O TEXTO) ---
# 1. Vamos criar uma tabela temporária
df_abuso = pd.DataFrame()
df_abuso['Bairro'] = df_limpo.loc[y_test.index, 'neighbourhood_cleansed']
df_abuso['Real'] = y_test.values
df_abuso['Justo'] = previsoes
df_abuso['Tipo'] = df_limpo.loc[y_test.index, 'room_type']
df_abuso['Pessoas'] = df_limpo.loc[y_test.index, 'accommodates']

# 2. Calculamos a Diferença (Real - Justo)
# Se o Real é 1000 e o Justo é 300, a diferença é +700 (Abuso!)
df_abuso['Diferenca'] = df_abuso['Real'] - df_abuso['Justo']

# 3. Ordenamos para pegar o MAIOR abuso de todos (o campeão da exploração)
top_abusivo = df_abuso.sort_values(by='Diferenca', ascending=False).iloc[0]

# 4. Calculamos a porcentagem de aumento
aumento_perc = ((top_abusivo['Real'] - top_abusivo['Justo']) / top_abusivo['Justo']) * 100

# 5. IMPRIMIR O TEXTO PRONTO PARA COPIAR E COLAR
print("\n" + "="*50)
print("AQUI ESTÁ O SEU EXEMPLO REAL (COPIE OS DADOS ABAIXO):")
print("="*50)
print(f"Bairro: {top_abusivo['Bairro']}")
print(f"Tipo: {top_abusivo['Tipo']}")
print(f"Capacidade: {top_abusivo['Pessoas']} pessoas")
print(f"Preço Real (Cobrado): R$ {top_abusivo['Real']:.2f}")
print(f"Preço Justo (IA): R$ {top_abusivo['Justo']:.2f}")
print(f"Diferença Absurda: R$ {top_abusivo['Diferenca']:.2f}")
print(f"Porcentagem de Abuso: +{aumento_perc:.0f}%")
print("="*50 + "\n")