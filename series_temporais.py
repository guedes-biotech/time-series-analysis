from statistics import mode, median, mean, variance, quantiles
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import kurtosis
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def coef_assim(numbers_list, method):
    standard_der = (variance(numbers_list))**(0.5)
    mean_val = mean(numbers_list)
    median_val = median(numbers_list)
    if method == 'pearson_mode':
        mode_val = mode(numbers_list)
        coef = (mean_val - mode_val) / standard_der
    if method == 'pearson_median':
        coef = (mean_val - median_val) / standard_der
    if method == "quartil":
        quartil_list = quantiles(numbers_list)
        coef = ((quartil_list[-1] + quartil_list[0]) - 2*median_val) / (quartil_list[-1] - quartil_list[0])
        
        
    return coef

df_data = pd.read_csv('M-IBV.csv')
df_data['IBOV'] = pd.to_numeric(df_data['IBOV'], errors='coerce')
df_data['Log_Retorno_IBOV'] = np.log(df_data['IBOV'] / df_data['IBOV'].shift(1))
log_retornos = df_data['Log_Retorno_IBOV'].dropna()
with open('Results.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['Medida', 'Valor'])
    writer.writerows([
        ['Média', mean(log_retornos)],
        ['Variância Amostral', variance(log_retornos)],
        ['Coef. assimetria', coef_assim(log_retornos, 'quartil')],
        ['Curtose', kurtosis(log_retornos, bias=False)],
        ['Máximo', max(log_retornos)],
        ['Mínimo', min(log_retornos)],
    ])
    size = len(log_retornos) - 1
    acf_values = acf(log_retornos, nlags = size)
    
    for i, value in enumerate(acf_values):
        writer.writerow([f'r{i}', value])
        
        
plt.figure(figsize=(12, 6))
plt.hist(log_retornos, bins=43, color='lightgrey', edgecolor='black')
plt.xlabel('Log-retornos mensais do IBOVESPA')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=18))
plt.ylabel('Frequência')
plt.savefig('Histograma.png', dpi=300, bbox_inches='tight')
