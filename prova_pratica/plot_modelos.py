import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

# Dados para plotagem
default_series = pd.read_csv("prova_pratica/11.csv")

# Crie uma figura
fig = plt.figure(figsize=(10, 8))

# Defina o layout dos gráficos
gs = gridspec.GridSpec(3, 2)

# Dados originais
ax1 = fig.add_subplot(gs[0, :])
ax1.set_title('Temperaturas superficiais')

ax1.plot(default_series["t"], default_series["x"], color="gray")

ax1.set_xticks([t for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])
ax1.set_xticklabels([str(t) for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])

ax1.set_xlabel('t (Mês)')

ax1.set_ylabel('Z(t) (Kelvin)')

# Médias Móveis Simples
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('MMS (Trimestre)')

ax2.plot(default_series["t"], default_series["x"], color="gray")
ax2.plot(default_series["x"].rolling(window=3).mean(), color="red", linestyle='dashed')

ax2.set_xticks([t for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])
ax2.set_xticklabels([str(t) for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])

# Suavização Exponencial Simples
modelo = SimpleExpSmoothing(default_series["x"]).fit(optimized=True)
alpha_otimo = modelo.model.params["smoothing_level"]

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title(f'SES (alpha = {round(alpha_otimo, 4)})')

ax3.plot(default_series["t"], default_series["x"], color="gray")
ax3.plot(default_series["t"], modelo.fittedvalues, color='red', linestyle='dashed')

ax3.set_xticks([t for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])
ax3.set_xticklabels([str(t) for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])

# Suavização exponencial de Holt
modelo = Holt(default_series["x"]).fit(optimized=True)
alpha_otimo = modelo.model.params["smoothing_level"]
beta_otimo = modelo.model.params["smoothing_trend"]

ax4 = fig.add_subplot(gs[2, 0])
ax4.set_title(f'SEH (A = {round(alpha_otimo, 4)}, C = {round(beta_otimo, 4)})')

ax4.plot(default_series["t"], default_series["x"], color="gray")
ax4.plot(default_series["t"], modelo.fittedvalues, color='red', linestyle='dashed')

ax4.set_xticks([t for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])
ax4.set_xticklabels([str(t) for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])

# Holt-Winters
modelo = ExponentialSmoothing(default_series["x"], trend="add", seasonal="add", seasonal_periods=12).fit(optimized=True)
alpha_otimo = modelo.model.params["smoothing_level"]
beta_otimo = modelo.model.params["smoothing_trend"]
gamma_otimo = modelo.model.params["smoothing_seasonal"]
print(alpha_otimo, beta_otimo, gamma_otimo)

ax5 = fig.add_subplot(gs[2, 1])
ax5.set_title(f'HW (A = {round(alpha_otimo, 4)}, C = {round(beta_otimo, 4)}, D = {round(gamma_otimo, 4)})')

ax5.plot(default_series["t"], default_series["x"], color="gray")
ax5.plot(default_series["t"], modelo.fittedvalues, color='red', linestyle='dashed')

ax5.set_xticks([t for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])
ax5.set_xticklabels([str(t) for t in default_series["t"] if t >= 7 and (t - 7) % 12 == 0])

# Ajuste o layout para evitar sobreposição de títulos
plt.tight_layout()

plt.savefig('prova_pratica/EX1')