import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ubatuba_full_series = pd.read_csv("Lista_2/temperatura.csv")

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1)

#Gráfico da série original
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Temperatura na Cidade de Ubatuba')

ax1.plot(ubatuba_full_series.index, ubatuba_full_series["Ubatuba"], color="gray")
jan_indices = range(0, len(ubatuba_full_series), 12)
jan_labels = [round(ubatuba_full_series['Ano'][i]) for i in jan_indices]

ax1.set_xticks(jan_indices)
ax1.set_xticklabels(jan_labels, rotation=90)

ax1.set_xlabel('Tempo')
ax1.set_ylabel('Temperatura')

#plot HW
modelo = ExponentialSmoothing(ubatuba_full_series["Ubatuba"], trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True)
alpha_otimo = modelo.model.params["smoothing_level"]
beta_otimo = modelo.model.params["smoothing_trend"]
gamma_otimo = modelo.model.params["smoothing_seasonal"]
print(alpha_otimo, beta_otimo, gamma_otimo)

ax2 = fig.add_subplot(gs[1, :])
ax2.set_title('Suavização da série utilizando Holt-Winters')

ax2.plot(ubatuba_full_series.index, ubatuba_full_series["Ubatuba"], color="gray")
ax2.plot(ubatuba_full_series.index, modelo.fittedvalues, color='red', linestyle='dashed')

ax2.set_xticks(jan_indices)
ax2.set_xticklabels(jan_labels, rotation=90)

plt.tight_layout()
plt.savefig("Lista_2/plot_ex2")
