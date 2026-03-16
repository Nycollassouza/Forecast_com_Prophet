📈 Forecast de Vendas com Prophet

Projeto de previsão de vendas utilizando Facebook Prophet, com suporte para múltiplos produtos, sazonalidade e geração automatizada de resultados.

O objetivo do projeto é criar um pipeline de previsão escalável, capaz de processar dados históricos de vendas e gerar projeções futuras de forma automatizada.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

🚀 Tecnologias utilizadas

Python
Prophet (Meta)
Pandas
NumPy
Scikit-learn
Matplotlib
YAML
Tkinter (interface)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 O que o projeto faz

Este projeto permite:

✔️ Treinar modelos de forecast com Prophet
✔️ Processar múltiplos produtos automaticamente
✔️ Considerar sazonalidade agrícola
✔️ Gerar previsões futuras
✔️ Exportar resultados para arquivos
✔️ Gerar gráficos de previsão

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

📂 Estrutura do projeto
Forecast_com_Prophet/

src/

├── main.py               
├── forecaster.py         
├── model_trainer.py       
├── output_generator.py    
├── sazonalidade_agro.py  
└── app_tk.py              

config/


├── config.yaml

├── pesos_distribuicao.yaml

└── sazonalidade_agro.yaml

data/

└── exemplo_base_forecast.xlsx

requirements.txt
README.md
.gitignore

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚙️ Instalação

Clone o repositório:

git clone https://github.com/Nycollassouza/Forecast_com_Prophet.git

Entre na pasta:

cd Forecast_com_Prophet

Crie um ambiente virtual:

python -m venv venv

Ative o ambiente:

Windows:

venv\Scripts\activate

Instale as dependências:

pip install -r requirements.txt

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

▶️ Como executar

Execute o arquivo principal:

python src/main.py

Ou utilize a interface gráfica:

python src/app_tk.py

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

📉 Exemplo de Forecast

O modelo utiliza Facebook Prophet para gerar previsões baseadas em:

Tendência

Sazonalidade

Histórico de vendas

O resultado inclui:

previsão (yhat)

limite inferior (yhat_lower)

limite superior (yhat_upper)

------------------------------------------

📊 Possíveis melhorias futuras

Dashboard interativo com Plotly

API para geração de forecasts

Pipeline automatizado

Integração com banco de dados

Deploy em cloud

-------------------------------------------

👨‍💻 Autor

Nycollas Faustino de Souza
