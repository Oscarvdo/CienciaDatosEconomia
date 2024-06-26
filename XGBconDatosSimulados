import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import xgboost as xgb
import numpy as np

# Datos simulados de análisis de sentimiento hacia TechCorp
data = {
    'Fecha': ['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04', '2023-06-05',
              '2023-06-06', '2023-06-07', '2023-06-08', '2023-06-09', '2023-06-10'],
    'Fuente': ['Twitter', 'Financial News', 'Twitter', 'Blog', 'Financial News',
               'Twitter', 'Financial News', 'Twitter', 'Financial News', 'Twitter'],
    'Texto': ['TechCorp announces a new partnership with a leading tech giant. Exciting times ahead! 😀',
              "TechCorp's quarterly earnings report exceeds expectations, showing strong growth in revenue and market share.",
              'Investors are skeptical about TechCorp\'s new product launch. Many concerns about its market acceptance. 😕',
              'An in-depth analysis of TechCorp\'s recent patent filings reveals potential for groundbreaking innovation in AI technology.',
              "TechCorp's CEO resigns unexpectedly amid rumors of internal conflicts and strategic changes.",
              "TechCorp's stock prices soar as analysts upgrade their ratings based on positive market sentiment. 🚀",
              'TechCorp faces a lawsuit over alleged patent infringement. Legal battle expected to impact stock prices.',
              'The tech community praises TechCorp\'s commitment to sustainability with their new green energy initiative. 👏',
              "TechCorp's latest acquisition strategy receives mixed reviews from industry experts. Uncertainty in market reaction.",
              "#TechCorp's AI-driven solutions are revolutionizing healthcare, improving patient outcomes globally. #HealthTech 🌐"],
    'Sentimiento': ['Positivo', 'Positivo', 'Negativo', 'Positivo', 'Negativo',
                    'Positivo', 'Negativo', 'Positivo', 'Neutral', 'Positivo']
}

# Convertir datos a DataFrame
df = pd.DataFrame(data)

# Convertir columna de fecha a formato datetime si es necesario
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Inicializar VADER para análisis de sentimiento
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Función para obtener la polaridad del sentimiento
def get_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positivo'
    elif scores['compound'] <= -0.05:
        return 'Negativo'
    else:
        return 'Neutral'

# Aplicar análisis de sentimiento
df['Sentimiento'] = df['Texto'].apply(get_sentiment)

# Generar puntuaciones numéricas para los sentimientos
sentiment_scores = {'Positivo': 1, 'Neutral': 0, 'Negativo': -1}
df['Sentimiento_Puntuacion'] = df['Sentimiento'].map(sentiment_scores)

# Agregar una columna con datos de precios de acciones (ficticios)
df['Precio_Accion'] = [150, 152, 148, 155, 145, 160, 140, 165, 158, 170]

# Codificar variables categóricas (por ejemplo, Fuente)
df = pd.get_dummies(df, columns=['Fuente'], drop_first=True)

# Variables predictoras y objetivo
X = df.drop(columns=['Fecha', 'Texto', 'Sentimiento', 'Precio_Accion'])
y = df['Precio_Accion']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de XGBoost
model = xgb.XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validación cruzada para MSE
cv_mse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_mse = np.mean(cv_mse)

# Validación cruzada para MAE
cv_mae = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
mean_cv_mae = np.mean(cv_mae)

# Predicción en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular MSE y MAE en el conjunto de prueba
test_mse = mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error Cuadrático Medio: {mse}')

# Resultados
print('Predicciones:', y_pred)
print('Valores Reales:', y_test.values)

# Imprimir resultados de validación cruzada y prueba
print(f'Error Cuadrático Medio (MSE) en Validación Cruzada: {mean_cv_mse}')
print(f'Error Absoluto Medio (MAE) en Validación Cruzada: {mean_cv_mae}')
print(f'Error Cuadrático Medio (MSE) en Conjunto de Prueba: {test_mse}')
print(f'Error Absoluto Medio (MAE) en Conjunto de Prueba: {test_mae}')
