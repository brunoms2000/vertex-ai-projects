import pandas as pd
import requests

df = pd.read_excel("base_atendimentos.xlsx", dtype=str)

url = 'urldacloudfunction'
df['Classificao Vertex'] = ''
for index, row in df.iterrows():
    chat = row['Transcricao']
    data_request = {"chat": chat}
    try:
        request = requests.post(url, json=data_request)
        text = request.text
        df.at[index, 'Classificao Vertex'] = text
    except requests.exceptions.RequestException as e:
        df.at[index, 'Classificao Vertex'] = f"Erro: Ocorreu um erro na requisição com a AI generativa: {e}."
        pass
print(df)
df.to_excel("base_atendimentos_resultado.xlsx", index=False)