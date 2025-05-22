import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Cargar y preparar datos
df = pd.read_csv('SI_L07_SVM_DATASET.csv')
df = df.drop(columns=['Rank', 'Country'])

# Codificar continentes
continent_mapping = {
    'Africa': 0,
    'Asia': 1,
    'Europe': 2,
    'North America': 3,
    'Oceania': 4,
    'South America': 5
}
df['Continent'] = df['Continent'].map(continent_mapping)

# Visualización de datos con pairplot
sns.pairplot(df, hue='Continent', palette='viridis', 
             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'})
plt.suptitle('Pairplot de Variables por Continente', y=1.02)
plt.show()

# Balancear clases con SMOTE
X = df.drop(columns=['Continent'])
y = df['Continent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Optimización de hiperparámetros
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_res, y_res)

best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

# Métricas de evaluación
print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Reporte de clasificación detallado
print("\nReporte de Clasificación Mejorado:")
print(classification_report(y_test, y_pred, target_names=continent_mapping.keys()))

# Matriz de confusión visual mejorada
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=continent_mapping.keys(), 
            yticklabels=continent_mapping.keys())
plt.title('Matriz de Confusión Mejorada')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Análisis de características (usando coeficientes para SVM lineal)
if best_svm.kernel == 'linear':
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_svm.coef_[0]
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Importancia de Características en SVM Lineal')
    plt.show()
else:
    print("Para kernels no lineales, considere usar métodos de permutación para importancia de características")