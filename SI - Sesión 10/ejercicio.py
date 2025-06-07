import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Cargar y mostrar datos originales sin etiquetar
print("="*50)
print("DATOS ORIGINALES SIN AGRUPAR")
print("="*50)

# Cargar datos
df = pd.read_csv('SI_L10_KMEANS_DATASET.csv')

# Seleccionar características relevantes
caracteristicas = ['wins', 'kills', 'kdRatio', 'level', 'scorePerMinute', 'gamesPlayed']
X = df[caracteristicas]

# Visualización de datos sin agrupar
plt.figure(figsize=(12, 6))
plt.scatter(X['wins'], X['kills'], alpha=0.6, color='blue')
plt.title('Distribución Original de Jugadores (sin agrupar)')
plt.xlabel('Número de Victorias')
plt.ylabel('Número de Asesinatos')
plt.grid(True)
plt.show()

# 2. Método del codo para determinar K óptimo
print("\n" + "="*50)
print("MÉTODO DEL CODO PARA DETERMINAR CLUSTERS ÓPTIMOS")
print("="*50)

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calcular WCSS para diferentes valores de K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Visualizar método del codo
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', color='green')
plt.title('Método del Codo para Determinar K Óptimo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# 3. Visualización de datos agrupados
print("\n" + "="*50)
print("RESULTADOS DEL CLUSTERING (K=5)")
print("="*50)

# Aplicar K-Means con K=5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualización de clusters
plt.figure(figsize=(12, 6))
plt.scatter(X['wins'], X['kills'], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:, 0], 
           scaler.inverse_transform(kmeans.cluster_centers_)[:, 1],
           s=200, c='red', marker='X', label='Centroides')
plt.title('Jugadores Agrupados por Rendimiento (5 clusters)')
plt.xlabel('Número de Victorias')
plt.ylabel('Número de Asesinatos')
plt.legend()
plt.grid(True)
plt.show()