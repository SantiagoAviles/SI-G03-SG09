import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# Configuramos el estilo de seaborn para mejores visualizaciones
sns.set_style('whitegrid')
sns.set_context('paper')

# Cargamos los datos del CSV
df = pd.read_csv('SI_L11_GMM_DATASET.csv')

# Calculamos métricas de volatilidad y volumen
df['rango_diario'] = df['high'] - df['low']
df['volatilidad_relativa'] = abs(df['close'] - df['open']) / df['open'] * 100
df['volumen_normalizado'] = df['volume'] / df['volume'].mean()

# Creamos una figura con dos subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

# Gráfico de precios
ax1.plot(df['date'], df['close'], label='Precio de Cierre', color='blue')
ax1.set_title('Precio de Cierre Histórico')
ax1.set_xlabel('')
ax1.set_ylabel('Precio')
ax1.legend()

# Gráfico de volumen normalizado
ax2.bar(df['date'], df['volumen_normalizado'], label='Volumen Normalizado', color='gray', alpha=0.5)
ax2.set_title('Volumen Normalizado')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Volumen (normalizado)')
ax2.legend()

plt.tight_layout()
plt.show()

# Preparamos las características para el análisis
caracteristicas = df[['rango_diario', 'volatilidad_relativa', 'volumen_normalizado']]

# Normalizamos las características usando Min-Max Scaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(caracteristicas)

# Calculamos BIC para diferentes números de componentes
n_components_range = range(2, 11)
bic_values = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(X_scaled)
    bic_values.append(gmm.bic(X_scaled))

# Visualizamos el BIC para cada número de componentes
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_values, marker='o')
plt.title('Criterio de Información Bayesiano (BIC)')
plt.xlabel('Número de Componentes')
plt.ylabel('BIC')
plt.grid(True)
plt.show()

# Encontramos el número óptimo de componentes (el que minimiza el BIC)
n_optimo = n_components_range[np.argmin(bic_values)]
print(f"\nNúmero óptimo de componentes según BIC: {n_optimo}")

# Creamos y entrenamos el modelo GMM con el número óptimo de componentes
gmm_optimo = GaussianMixture(n_components=n_optimo, covariance_type='full', random_state=42)
gmm_optimo.fit(X_scaled)

# Obtenemos las etiquetas de clustering para cada día
df['cluster'] = gmm_optimo.predict(X_scaled)

# Visualizamos los clusters en un espacio 2D usando las dos primeras características
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.title('Clustering de Días de Negociación')
plt.xlabel('Rango Diario Normalizado')
plt.ylabel('Volatilidad Relativa Normalizada')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Calculamos estadísticas por cluster
estadisticas_cluster = df.groupby('cluster').agg({
    'rango_diario': ['mean', 'std'],
    'volatilidad_relativa': ['mean', 'std'],
    'volumen_normalizado': ['mean', 'std'],
    'cluster': 'count'
}).round(2)

print("\nEstadísticas por Cluster:")
print(estadisticas_cluster)

# Calculamos las probabilidades de pertenencia a cada cluster
probabilidades = gmm_optimo.predict_proba(X_scaled)

# Creamos un gráfico que muestre las probabilidades de pertenencia
plt.figure(figsize=(12, 6))
for i in range(n_optimo):
    plt.plot(df['date'], probabilidades[:, i], label=f'Cluster {i}', alpha=0.7)
plt.title('Probabilidad de Pertenencia a Cada Cluster')
plt.xlabel('Fecha')
plt.ylabel('Probabilidad')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Creamos un gráfico que combine precio, volumen y clusters
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

# Gráfico de precio con colores según cluster
scatter = ax1.scatter(df['date'], df['close'], c=df['cluster'], cmap='viridis')
ax1.set_title('Precio de Cierre por Cluster')
ax1.set_xlabel('')
ax1.set_ylabel('Precio')
plt.colorbar(scatter, ax=ax1, label='Cluster')

# Gráfico de volumen con colores según cluster
scatter2 = ax2.scatter(df['date'], df['volumen_normalizado'], c=df['cluster'], cmap='viridis')
ax2.set_title('Volumen Normalizado por Cluster')
ax2.set_xlabel('Fecha')
ax2.set_ylabel('Volumen (normalizado)')
plt.colorbar(scatter2, ax=ax2, label='Cluster')

plt.tight_layout()
plt.show()

# Calculamos las características promedio de cada cluster
caracteristicas_cluster = df.groupby('cluster')[['rango_diario', 'volatilidad_relativa', 'volumen_normalizado']].mean().round(2)
print("\nCaracterísticas Promedio por Cluster:")
print(caracteristicas_cluster)

# Calculamos la proporción de días en cada cluster
proporcion_cluster = df['cluster'].value_counts(normalize=True).round(4) * 100
print("\nProporción de Días en Cada Cluster (%):")
print(proporcion_cluster)