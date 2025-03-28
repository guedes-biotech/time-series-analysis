import pandas as pd
from statsmodels.tsa.holtwinters import Holt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

oleo_full_series = pd.read_csv("Lista_2/venda_oleo.csv")
train_series = oleo_full_series.iloc[:-12].copy()
para_prever = oleo_full_series.iloc[-12: -5].copy()

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1)

#Gráfico da série original
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Venda mensal de óleo lubrificante')

ax1.plot(oleo_full_series["mes_ano"], oleo_full_series["vendas"], color="gray")

jan_indices = [i for i, mes in enumerate(oleo_full_series['mes_ano']) if 'jan' in mes]
jan_labels = [oleo_full_series['mes_ano'][i] for i in jan_indices]
ax1.set_xticks(jan_indices)
ax1.set_xticklabels(jan_labels, rotation=90)

ax1.set_xlabel('Mês/Ano')
ax1.set_ylabel('vendas')

#Previsão do modelo
modelo = Holt(train_series["vendas"]).fit(optimized=True)
alpha_otimo = modelo.model.params["smoothing_level"]
beta_otimo = modelo.model.params["smoothing_trend"]

ax2 = fig.add_subplot(gs[1, :])
ax2.set_title('Série suavizada até dezembro de 1977')
ax2.plot(train_series["mes_ano"], modelo.fittedvalues, color='red', linestyle='dashed')
ax2.plot(oleo_full_series["mes_ano"], oleo_full_series["vendas"], color="gray")
jan_indices = [i for i, mes in enumerate(oleo_full_series['mes_ano']) if 'jan' in mes]
jan_labels = [oleo_full_series['mes_ano'][i] for i in jan_indices]
ax2.set_xticks(jan_indices)
ax2.set_xticklabels(jan_labels, rotation=90)

previsao_sem_at = modelo.forecast(7)

previsao_sem_at.to_csv('Lista_2/resultados_4_sem_atualizacao.csv')

forecast_results = pd.DataFrame(columns=["index", "previsao"])
for _ in range(7):
    index_real = train_series.index.max() + 1
    valor_real = oleo_full_series.loc[index_real, "vendas"]
    
    previsao = modelo.forecast(1)
    forecast_results = pd.concat([forecast_results, pd.DataFrame({"index": [index_real], "previsao": [previsao.iloc[0]]})], ignore_index=True)
    train_series = pd.concat([train_series, pd.DataFrame({'mes_ano': [index_real], 'vendas': [valor_real]})], ignore_index=True) 
    modelo = Holt(train_series["vendas"]).fit(optimized=True)
    
forecast_results.to_csv('Lista_2/resultados_4_com_atualizacao.csv')

para_prever = para_prever.reset_index(drop=True)
forecast_results = forecast_results.reset_index(drop=True)

diff_df = pd.DataFrame()
diff_df['sem_atualizacao'] = para_prever["vendas"] - previsao_sem_at.values
diff_df['com_atualizacao'] = para_prever["vendas"].astype(float) - forecast_results["previsao"]
ax3 = fig.add_subplot(gs[2, :])
ax3.set_title('erros de previsão por valor da série')
ax3.scatter(para_prever["vendas"], diff_df['com_atualizacao'], color='green', marker='x')
ax3.axhline(y=0, color='black', linestyle='--')

ax3.set_xlabel('vendas')
ax3.set_ylabel('erro')

plt.tight_layout()
plt.savefig("Lista_2/plot_ex4")