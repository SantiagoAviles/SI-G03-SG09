import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
data = pd.read_csv('tumores.csv')

# Verificar valores faltantes
print("Valores faltantes por columna:")
print(data.isnull().sum())

# Ver distribución de clases
print("\nDistribución de clases:")
print(data['Class'].value_counts())

# Separar características y variable objetivo
X = data.drop('Class', axis=1)
y = data['Class']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train)

# Hacer predicciones
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probabilidades para clase 1 (benigno)

# Métricas de evaluación
print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Visualización de la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()

# Importancia de las características
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Absolute_Coefficient': np.abs(model.coef_[0])
}).sort_values('Absolute_Coefficient', ascending=False)

print("\nImportancia de las características:")
print(feature_importance)

# Visualización de la importancia
plt.figure(figsize=(10, 6))
sns.barplot(x='Absolute_Coefficient', y='Feature', data=feature_importance)
plt.title('Importancia de las Características (Valor Absoluto de Coeficientes)')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.show()
