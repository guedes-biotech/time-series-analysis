import pandas as pd
from statsmodels.tsa.holtwinters import Holt

# Dados para plotagem
full_series = pd.read_csv("prova_pratica/11.csv")
uncomplete_serie = full_series.iloc[:-3].copy()


modelo = Holt(uncomplete_serie["x"]).fit(optimized=True)
previsao = modelo.forecast(3)
previsao.to_csv('prova_pratica/previsao_sem_atualizacao.csv', index=False)

forecast_results = pd.DataFrame(columns=["t", "previsao"])
for _ in range(3):
    t_real = uncomplete_serie["t"].max() + 1
    valor_real = full_series.loc[full_series["t"] == t_real, "x"].values[0]
    
    previsao = modelo.forecast(1)
    forecast_results = pd.concat([forecast_results, pd.DataFrame({"t": [t_real], "previsao": [previsao.iloc[0]]})], ignore_index=True)
    uncomplete_serie = pd.concat([uncomplete_serie, pd.DataFrame({'t': [t_real], 'x': [valor_real]})], ignore_index=True) 
    modelo = Holt(uncomplete_serie["x"]).fit(optimized=True)
    
forecast_results.to_csv('prova_pratica/previsao_com_atualizacao.csv')