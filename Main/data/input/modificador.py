import pandas as pd

df = pd.read_excel(r'C:\Users\SouZ\Documents\Modelo de Previsão Prophet\venv\data\input\HIstorico_2023_2024.xlsx')
df.to_csv(r'C:\Users\SouZ\Documents\Modelo de Previsão Prophet\venv\data\input\HIstorico_2023_2024.csv', index=False)