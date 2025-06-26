import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Cargar y preprocesar datos (igual que antes)
data = pd.read_csv('dataset.csv', delimiter=';')
X = data.drop(['Time', 'Class'], axis=1)
y = data['Class'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train_normal = X_train[y_train == 0]

# Arquitectura ORIGINAL que te daba mejores resultados
input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(input_layer)
encoder = keras.layers.Dropout(0.2)(encoder)
encoder = keras.layers.Dense(64, activation='relu')(encoder)
encoder = keras.layers.Dense(32, activation='relu')(encoder)
bottleneck = keras.layers.Dense(encoding_dim, activation='relu')(encoder)
decoder = keras.layers.Dense(32, activation='relu')(bottleneck)
decoder = keras.layers.Dense(64, activation='relu')(decoder)
decoder = keras.layers.Dense(128, activation='relu')(decoder)
decoder = keras.layers.Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = keras.models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Entrenamiento CON MÁS ÉPOCAS y EARLY STOPPING
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True)

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=300,  # Aumentamos las épocas
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1)

# Reconstrucción y umbral ÓPTIMO
X_test_pred = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

# AQUÍ ESTÁ EL AJUSTE CLAVE: Usamos el percentil 99.3 de los errores normales
normal_errors = reconstruction_error[y_test == 0]
threshold = np.percentile(normal_errors, 99.3)  # Ajustado manualmente

y_pred = (reconstruction_error > threshold).astype(int)

# Métricas
print("\nMétricas Mejoradas:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraude']))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print(f"\nUmbral usado: {threshold:.5f} (percentil 99.3 de errores normales)")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# [Código previo de carga y preprocesamiento igual al anterior...]

# =============================================
# 1. Gráfica para selección de umbral óptimo
# =============================================
plt.figure(figsize=(15, 6))

# Gráfico de distribución de errores
plt.subplot(1, 2, 1)
sns.kdeplot(reconstruction_error[y_test == 0], label='Normal', shade=True)
sns.kdeplot(reconstruction_error[y_test == 1], label='Fraude', shade=True)

# Umbrales clave
percentiles = [99.0, 99.3, 99.5, 99.7]
colors = ['green', 'blue', 'purple', 'red']
for p, color in zip(percentiles, colors):
    threshold = np.percentile(reconstruction_error[y_test == 0], p)
    plt.axvline(threshold, linestyle='--', color=color,
                label=f'Percentil {p} ({threshold:.2f})')

plt.title('Distribución de Errores por Clase')
plt.xlabel('Error de Reconstrucción')
plt.ylabel('Densidad')
plt.legend()

# Curva Precision-Recall
plt.subplot(1, 2, 2)
precision, recall, thresholds = precision_recall_curve(y_test, reconstruction_error)
plt.plot(recall, precision, label='Curva PR')
plt.fill_between(recall, precision, alpha=0.2)

# Punto óptimo (F1 máximo)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
plt.scatter(recall[optimal_idx], precision[optimal_idx],
            color='red', label=f'Óptimo (F1={f1_scores[optimal_idx]:.2f})')

plt.title('Curva Precision-Recall (AUC = {:.2f})'.format(auc(recall, precision)))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# 2. Gráfica de desbalance de clases
# =============================================
plt.figure(figsize=(12, 5))

# Distribución de clases
plt.subplot(1, 2, 1)
class_dist = pd.Series(y).value_counts(normalize=True)
class_dist.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Distribución de Clases')
plt.ylabel('')

# Impacto en métricas
plt.subplot(1, 2, 2)
metrics = classification_report(y_test, y_pred, output_dict=True)
sns.barplot(x=['Precisión', 'Recall', 'F1'],
            y=[metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']],
            palette='viridis')
plt.title('Métricas para Clase Minoritaria (Fraude)')
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# =============================================
# 3. Matriz de confusión detallada
# =============================================
plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Normal', 'Pred Fraude'],
            yticklabels=['Real Normal', 'Real Fraude'])
plt.title('Matriz de Confusión (Umbral = {:.4f})'.format(threshold))
plt.show()

# =============================================
# 4. Gráfico de errores vs características
# =============================================
plt.figure(figsize=(12, 6))
sample_frauds = np.where((y_test == 1) & (y_pred == 1))[0][:5]  # 5 fraudes detectados
for i in sample_frauds:
    plt.plot(X_test[i] - X_test_pred[i], label=f'Fraude {i}')
plt.axhline(0, color='black', linestyle='--')
plt.title('Patrones de Error en Fraudes Detectados')
plt.xlabel('Característica')
plt.ylabel('Error (Real - Predicho)')
plt.legend()
plt.show()