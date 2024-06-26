# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Datos de ejemplo (simulados)
data = {
    'Fecha': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Precio Apertura': [150.25, 152.60, 153.00, 151.50, 152.75],
    'Precio Cierre': [152.50, 153.20, 151.80, 153.00, 154.00],
    'Volumen Operaciones': [2500000, 2300000, 2700000, 2200000, 2400000],
    'P/E': [25.6, 26.0, 25.8, 26.2, 25.5],
    'P/B': [3.2, 3.1, 3.3, 3.0, 3.2],
    'Sentimiento Noticias': ['Positivo', 'Neutral', 'Negativo', 'Positivo', 'Neutral']
}

# Convertir a DataFrame de pandas
df = pd.DataFrame(data)

# Preparar variables predictoras y variable objetivo
X = df[['Precio Apertura', 'Volumen Operaciones', 'P/E', 'P/B']]
y = df['Precio Cierre']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de Bosques Aleatorios
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Obtener predicciones del conjunto de prueba
predictions = model.predict(X_test)

# Evaluar el modelo utilizando validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores  # Convertir a valores positivos
cv_mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

# Calcular el error medio cuadrático (MSE) y el error absoluto medio (MAE)
mse = mean_squared_error(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))

# Imprimir resultados
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Validación Cruzada - MSE: {cv_mse_scores.mean():.2f}')
print(f'Validación Cruzada - MAE: {cv_mae_scores.mean():.2f}')

# Mostrar las primeras 10 predicciones como ejemplo
print("\nEjemplo de predicciones:")
for i, prediction in enumerate(predictions[:10]):
    print(f'Predicción {i+1}: {prediction:.2f}')
