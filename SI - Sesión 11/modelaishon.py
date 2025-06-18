import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Cargar los datos
data = pd.read_csv('SI_L11_GMM_DATASET.csv')

# Calcular características relevantes
data['rango_diario'] = (data['high'] - data['low']) / data['open'] * 100  # Volatilidad porcentual
data['retorno'] = (data['close'] - data['open']) / data['open'] * 100  # Retorno porcentual diario

# Seleccionar características para el clustering
features = data[['rango_diario', 'retorno', 'volume']]

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(features)
# Encontrar el número óptimo de clusters usando BIC
bic_scores = []
n_components = range(1, 10)

for n in n_components:
    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
    gmm.fit(X)
    bic_scores.append(gmm.bic(X))

# Graficar BIC
plt.plot(n_components, bic_scores, marker='o', linestyle='-')
plt.xlabel("Número de Clusters")
plt.ylabel("BIC Score")
plt.title("Método de BIC para Gaussian Mixture Model")
plt.show()
# Aplicar GMM con el número óptimo de clusters
optimal_clusters = 4  # Ajustar según gráfico BIC
gmm = GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# Añadir etiquetas al dataframe original
data['cluster'] = labels
plt.figure(figsize=(10, 6))
plt.scatter(data['rango_diario'], data['retorno'], c=data['cluster'], cmap='viridis', alpha=0.6)
plt.xlabel("Volatilidad Diaria (% rango)")
plt.ylabel("Retorno Diario (%)")
plt.title("Segmentación de Días de Negociación con GMM")
plt.colorbar(label='Cluster')
plt.show()
# Analizar características por cluster
cluster_stats = data.groupby('cluster')[['rango_diario', 'retorno', 'volume']].describe()

# Mostrar estadísticas clave
print(cluster_stats.loc[:, (slice(None), ['mean', 'std'])])

# Interpretación de clusters
for cluster in range(optimal_clusters):
    cluster_data = data[data['cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"- Días: {len(cluster_data)} ({len(cluster_data)/len(data)*100:.1f}%)")
    print(f"- Volatilidad media: {cluster_data['rango_diario'].mean():.2f}%")
    print(f"- Retorno medio: {cluster_data['retorno'].mean():.2f}%")
    print(f"- Volumen medio: {cluster_data['volume'].mean():.0f}")

