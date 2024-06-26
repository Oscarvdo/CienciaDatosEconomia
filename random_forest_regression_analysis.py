# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cargar datos desde un archivo CSV
file_path = 'ruta/al/archivo/datosNoticias.csv'  # Reemplazar con la ruta correcta del archivo CSV
df = pd.read_csv(file_path)

# Convertir la fecha a formato datetime si es necesario
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Codificar variables categóricas (Sentimiento Noticias) si es necesario
df = pd.get_dummies(df, columns=['Sentimiento Noticias'], drop_first=True)

# Dividir los datos en variables predictoras (X) y variable objetivo (y)
X = df.drop(['Fecha', 'Precio Cierre'], axis=1)
y = df['Precio Cierre']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de Bosques Aleatorios
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar el modelo utilizando validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores  # Convertir a valores positivos
cv_mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')

# Calcular el error medio cuadrático (MSE) y el error absoluto medio (MAE) en el conjunto de prueba
mse = mean_squared_error(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))

# Imprimir resultados
print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
print(f'Error Absoluto Medio (MAE): {mae:.2f}')
print(f'Validación Cruzada - MSE: {cv_mse_scores.mean():.2f}')
print(f'Validación Cruzada - MAE: {cv_mae_scores.mean():.2f}')
