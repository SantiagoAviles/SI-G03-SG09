import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Cargar los datos
data = pd.read_csv('diabetes.csv')

# Exploración inicial
print(data.head())
print(data.info())
print(data.describe())

# Distribución de la variable objetivo
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=data)
plt.title('Distribución de Diabetes (0=No, 1=Sí)')
plt.show()

# Relación entre variables
sns.pairplot(data, hue='Outcome', vars=['Glucose', 'BMI', 'Age', 'BloodPressure'])
plt.show()

# Matriz de correlación
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Separar características y variable objetivo
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Probar diferentes valores de K
k_values = range(20, 120)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Graficar precisión vs K
plt.figure(figsize=(10,6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('Número de Vecinos (K)')
plt.ylabel('Precisión')
plt.title('Precisión del Modelo KNN para Diferentes Valores de K')
plt.xticks(k_values)
plt.grid()
plt.show()

# Seleccionar el mejor K
best_k = k_values[np.argmax(accuracies)]
print(f"El mejor valor de K encontrado: {best_k}")

# Entrenar modelo con mejor K
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)

# Predicciones
y_pred = knn_best.predict(X_test)

# Evaluación
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Precisión del modelo con K={best_k}: {accuracy:.2f}')
print("Matriz de Confusión:\n", conf_matrix)
print("Reporte de Clasificación:\n", class_report)

# Validación cruzada para mejor estimación
cv_scores = cross_val_score(knn_best, X_train, y_train, cv=5)
print(f"Precisión promedio con validación cruzada: {np.mean(cv_scores):.2f}")

# Visualización de predicciones correctas/incorrectas
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_test[:,1], y=X_test[:,5], hue=y_test==y_pred, palette=['red','green'])
plt.xlabel('Glucose (estandarizada)')
plt.ylabel('BMI (estandarizado)')
plt.title('Predicciones Correctas (Verde) vs Incorrectas (Rojo)')
plt.legend(title='Predicción', labels=['Incorrecta', 'Correcta'])
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Modelos a comparar
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k)
}

# Evaluar cada modelo
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Mostrar resultados
print("\nComparación de Modelos:")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}")

# Gráfico de comparación de modelos
plt.figure(figsize=(10,6))
plt.bar(results.keys(), results.values())
plt.ylim(0.6, 1.0)
plt.ylabel('Precisión')
plt.title('Comparación de Diferentes Algoritmos de Clasificación')
plt.xticks(rotation=45)
plt.show()