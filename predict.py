import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 1. Загрузка данных (Берем BTC-USD за последние 5 лет)
print("Загружаю данные, не торопись...")
data = yf.download('BTC-USD', start='2019-01-01', end=None)
df = data[['Close']]

# 2. Подготовка данных
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

prediction_days = 60 # На сколько дней назад смотрит ИИ, чтобы сделать прогноз

x_train, y_train = [], []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 3. Создание нейросети (LSTM)
model = Sequential()

# Слой 1
model.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Защита от переобучения
# Слой 2
model.add(LSTM(units=60, return_sequences=True))
model.add(Dropout(0.2))
# Слой 3
model.add(LSTM(units=60))
model.add(Dropout(0.2))

model.add(Dense(units=1)) # Финальный прогноз цены

model.compile(optimizer='adam', loss='mean_squared_error')
print("Начинаю обучение нейронов... Это может занять время.")
model.fit(x_train, y_train, epochs=25, batch_size=32)

# 4. Прогноз на завтра
test_data = yf.download('BTC-USD', start='2023-01-01', end=None)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((df['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(f"\n" + "="*30)
print(f"ПРОГНОЗ ЦЕНЫ BTC НА СЛЕДУЮЩИЙ ДЕНЬ: {prediction[0][0]:.2f} USD")
print("="*30)

# 5. Визуализация (чтобы ты видел, как ИИ попадает в график)
test_prediction = model.predict(x_train) # Упрощенно для визуализации
test_prediction = scaler.inverse_transform(test_prediction)

plt.figure(figsize=(12,6))
plt.plot(df.index[prediction_days:], df['Close'][prediction_days:], color="black", label="Реальность")
plt.plot(df.index[prediction_days:], test_prediction, color="green", label="Предсказание ИИ")
plt.title("BTC Price Prediction - God Mode")
plt.xlabel("Время")
plt.ylabel("Цена ($)")
plt.legend()
plt.show()