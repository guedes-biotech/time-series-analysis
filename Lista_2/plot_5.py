import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

bebida_full_series = pd.read_csv("Lista_2/bebida.csv")
train_series = bebida_full_series.iloc[:180].copy()
para_prever = bebida_full_series.iloc[180:].copy()

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1)

#Gráfico da série original
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Indice-Bebidas')

ax1.plot(bebida_full_series.index, bebida_full_series["indice"], color="gray")

ax1.set_xticks(range(0, len(bebida_full_series), 12))
ax1.set_xlabel('t')
ax1.set_ylabel('Indice de Produção')

#plot HW 
ax2 = fig.add_subplot(gs[1, :])
ax2.set_title('Suavização Holt-Winters')

ax2.plot(train_series.index, train_series["indice"], color="gray")
modelo = ExponentialSmoothing(train_series["indice"], trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)
ax2.plot(train_series.index, modelo.fittedvalues, color='red', linestyle='dashed')

ax2.set_xticks(range(0, len(train_series), 12))

#plot previsão
ax3 = fig.add_subplot(gs[2, :])
ax3.set_title('Previsão - Ano de 2000')

ax3.plot(para_prever.index, para_prever["indice"], color="gray")
previsao = modelo.forecast(7)
ax3.plot(range(180, 187), previsao, color="green",linestyle='dashed')
previsao_df = pd.DataFrame(columns=['indice'])

for x in range(7):
    previsao = modelo.forecast(1)
    previsao_df = previsao_df._append({'indice': previsao.iloc[0]}, ignore_index=True)
    
    train_series = train_series._append(para_prever.iloc[x])
    modelo = ExponentialSmoothing(train_series["indice"], trend="add", seasonal="mul", seasonal_periods=12).fit(optimized=True)

ax3.plot(range(180, 187), previsao_df['indice'], color="blue", linestyle='dashed')

ax3.set_xticks(range(180, 187))

plt.tight_layout()
plt.savefig("Lista_2/plot_ex5")