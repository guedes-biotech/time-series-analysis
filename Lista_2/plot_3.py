import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

vendas_full_series = pd.read_csv("Lista_2/vendas.csv")
vendas_train = vendas_full_series.iloc[:-1].copy()

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1)

#Gráfico da série original
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Vendas Mensais')

ax1.plot(vendas_full_series['mes_ano'], vendas_full_series["vendas"], color="gray")

ax1.set_xticks(vendas_full_series['mes_ano'])
ax1.set_xticklabels(vendas_full_series['mes_ano'], rotation=90)

ax1.set_xlabel('mês/ano')
ax1.set_ylabel('Nº de vendas')

#plot medias moveis
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Vendas Suavizada por Médias Móveis')

ax2.plot(vendas_full_series["mes_ano"], vendas_full_series["vendas"], color="gray")

r3 = vendas_full_series["vendas"].rolling(window=3).mean()
r5 = vendas_full_series["vendas"].rolling(window=5).mean()
r9 = vendas_full_series["vendas"].rolling(window=9).mean()
ax2.plot(r3, color="red", linestyle='dashed')
ax2.plot(r5, color="green", linestyle='dashed')
ax2.plot(r9, color="blue", linestyle='dashed')
print(f"r = 3: {r3.dropna().iloc[-1]}")
print(f"\tabr-76: {r3.dropna().iloc[-2]}")
print(f"r = 5: {r5.dropna().iloc[-1]}")
print(f"\tabr-76: {r5.dropna().iloc[-2]}")
print(f"r = 9: {r9.dropna().iloc[-1]}")
print(f"\tabr-76: {r9.dropna().iloc[-2]}")

ax2.set_xticks(vendas_full_series['mes_ano'])
ax2.set_xticklabels(vendas_full_series['mes_ano'], rotation=90)

#plot suavizacao exponencial
ax3 = fig.add_subplot(gs[2, 0])

ax3.set_title('Vendas com Suavização Exponencial')

ax3.plot(vendas_full_series["mes_ano"], vendas_full_series["vendas"], color="gray")

modelo = SimpleExpSmoothing(vendas_full_series["vendas"]).fit(smoothing_level=0.1, optimized=False)
ax3.plot(vendas_full_series["mes_ano"], modelo.fittedvalues, color='red', linestyle='dashed')
previsao = modelo.forecast(1)
print("a=0.1: ",previsao.iloc[0])

modelo = SimpleExpSmoothing(vendas_train["vendas"]).fit(smoothing_level=0.1, optimized=False)
previsao = modelo.forecast(1)
print(f'\tabr-76: {previsao.iloc[0]}')

modelo = SimpleExpSmoothing(vendas_full_series["vendas"]).fit(smoothing_level=0.3, optimized=False)
ax3.plot(vendas_full_series["mes_ano"], modelo.fittedvalues, color='green', linestyle='dashed')
previsao = modelo.forecast(1)
print("a=0.3: ", previsao.iloc[0])

modelo = SimpleExpSmoothing(vendas_train["vendas"]).fit(smoothing_level=0.3, optimized=False)
previsao = modelo.forecast(1)
print(f'\tabr-76: {previsao.iloc[0]}')

modelo = SimpleExpSmoothing(vendas_full_series["vendas"]).fit(smoothing_level=0.7, optimized=False)
ax3.plot(vendas_full_series["mes_ano"], modelo.fittedvalues, color='blue', linestyle='dashed')
previsao = modelo.forecast(1)
print("a=0.7: ",previsao.iloc[0])

modelo = SimpleExpSmoothing(vendas_train["vendas"]).fit(smoothing_level=0.7, optimized=False)
previsao = modelo.forecast(1)
print(f'\tabr-76: {previsao.iloc[0]}')

ax3.set_xticks(vendas_full_series['mes_ano'])
ax3.set_xticklabels(vendas_full_series['mes_ano'], rotation=90)

plt.tight_layout()
plt.savefig("Lista_2/plot_ex3")