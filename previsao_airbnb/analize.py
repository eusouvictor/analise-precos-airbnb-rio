import os
import pandas as pd

# Caminho absoluto para o arquivo
file_path = os.path.join(os.getcwd(), 'previsao_airbnb', 'listings.csv.gz')
if not os.path.exists(file_path):
    raise FileNotFoundError(f"O arquivo '{file_path}' não foi encontrado. Certifique-se de que ele está no local correto.")

df = pd.read_csv(file_path)

print(df.head())