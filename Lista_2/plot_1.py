import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress

micv_full_series = pd.read_csv("Lista_2/ICV.csv")

# Converter a coluna de tempo para um índice numérico
tempo = np.arange(len(micv_full_series))
Zt = micv_full_series["ICV"].values

# Gráfico da série original
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1)

ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('M-ICV')
ax1.plot(micv_full_series["Mes/ano"], micv_full_series["ICV"], color="gray")

jan_indices = [i for i, mes in enumerate(micv_full_series['Mes/ano']) if 'Jan' in mes]
jan_labels = [micv_full_series['Mes/ano'][i] for i in jan_indices]
ax1.set_xticks(jan_indices)
ax1.set_xticklabels(jan_labels, rotation=90)
ax1.set_xlabel('Mês/Ano')
ax1.set_ylabel('Z(t)')

#plot medias moveis
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('M-ICV suavizada por médias móveis')

ax2.plot(micv_full_series["Mes/ano"], micv_full_series["ICV"], color="gray")
ax2.plot(micv_full_series["ICV"].rolling(window=3).mean(), color="red", linestyle='dashed')
ax2.set_xticks(jan_indices)
ax2.set_xticklabels(jan_labels, rotation=90)

#plot primeira diferença
ax3 = fig.add_subplot(gs[2, 0])
ax3.set_title('M-ICV primeira diferença')

micv_full_series['diferença'] = 0
for t in range(1, len(micv_full_series)):
    micv_full_series.loc[t, 'diferença'] = micv_full_series.loc[t, "ICV"] - micv_full_series.loc[t-1, "ICV"]
ax3.plot(micv_full_series["Mes/ano"], micv_full_series["diferença"], color="red", linestyle='dashed')
ax3.set_xticks(jan_indices)
ax3.set_xticklabels(jan_labels, rotation=90)

plt.tight_layout()
plt.savefig("Lista_2/plot_ex1")

#teste de sequências
diffs = np.diff(Zt)
sinais = np.sign(diffs)
n_trocas = np.sum(np.diff(sinais) != 0)
print(f"Número de trocas de sinal: {n_trocas}")


# Tt coeficientes
log_Zt = np.log(Zt)
slope, intercept, _, _, _ = linregress(tempo, log_Zt)
beta0 = np.exp(intercept)
beta1 = slope
print(f"Estimativa: beta0 = {beta0:.4f}, beta1 = {beta1:.4f}")
modelo_Tt = lambda t: beta0 * np.exp(beta1 * t)

# Previsões para 07/80 e 08/80
t_novos = [126, 127]
previsoes = [modelo_Tt(t) for t in t_novos]
print(f"Previsões para 07/80 e 08/80: {previsoes}")
