import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Cargar y explorar datos
data = pd.read_csv('calidadvino.csv', sep=';')
print("=== Primeras filas del dataset ===")
print(data.head())
print("\n=== Resumen estadístico ===")
print(data.describe())
print("\n=== Distribución de calidades ===")
print(data['quality'].value_counts().sort_index())

# 2. Visualización de datos
plt.figure(figsize=(15, 10))

# Distribución de calidad
plt.subplot(2, 2, 1)
sns.countplot(x='quality', data=data)
plt.title('Distribución de Calidad Original')

# Convertir a clases
data['quality_class'] = pd.cut(data['quality'], bins=[2,4,6,9], labels=[0,1,2])

# Distribución de clases
plt.subplot(2, 2, 2)
sns.countplot(x='quality_class', data=data)
plt.title('Distribución de Clases (Baja/Media/Alta)')

# Boxplots de características vs calidad
plt.subplot(2, 2, 3)
sns.boxplot(x='quality_class', y='alcohol', data=data)
plt.title('Alcohol vs Calidad')

plt.subplot(2, 2, 4)
sns.boxplot(x='quality_class', y='volatile acidity', data=data)
plt.title('Acidez Volátil vs Calidad')

plt.tight_layout()
plt.show()

# Matriz de correlación
plt.figure(figsize=(12, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.show()

# 3. Preparación de datos
X = data.drop(['quality', 'quality_class'], axis=1)
y = data['quality_class']

# Escalamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Modelado con Árbol de Decisión
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Árbol de Decisión ===")
print(f"Precisión: {accuracy_score(y_test, y_pred_dt):.2f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred_dt))

# Matriz de confusión
cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8,6))
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Baja", "Media", "Alta"],
            yticklabels=["Baja", "Media", "Alta"])
plt.title("Matriz de Confusión - Árbol de Decisión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# Visualización del árbol
plt.figure(figsize=(25,15))
plot_tree(dt, feature_names=X.columns, 
          class_names=["Baja", "Media", "Alta"], 
          filled=True, rounded=True, proportion=True, fontsize=10)
plt.title("Árbol de Decisión para Clasificación de Vino", fontsize=16)
plt.show()

# Importancia de características
dt_importance = pd.DataFrame({'Feature': X.columns, 'Importance': dt.feature_importances_})
dt_importance = dt_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=dt_importance)
plt.title('Importancia de Variables - Árbol de Decisión')
plt.show()

