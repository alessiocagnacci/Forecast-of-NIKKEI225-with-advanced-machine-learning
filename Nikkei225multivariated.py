#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE E RACCOLTA DATI MULTIVARIATI ---
# ------------------------------------------------------
print("1. Configurazione e Raccolta Dati (Nikkei + S&P 500)...")

# Variabili di configurazione
TICKER_NIKKEI = "^N225"
TICKER_SP500 = "^GSPC" # Variabile Esogena
START_DATE = "2015-01-01"

# Parametri del Modello Ottimizzati
LOOKBACK = 120  
TRAIN_SPLIT = 0.8 
FUTURE_DAYS = 20 

# 1.1 Download dei dati
data_nikkei = yf.download(TICKER_NIKKEI, start=START_DATE)
data_sp500 = yf.download(TICKER_SP500, start=START_DATE)

# 1.2 Selezione e Allineamento (Merge)
# Renaming per evitare conflitti e selezionare solo il prezzo di chiusura
df_nikkei = data_nikkei[['Close']].rename(columns={'Close': 'Nikkei_Close'})
df_sp500 = data_sp500[['Close']].rename(columns={'Close': 'SP500_Close'})

# Unione dei due dataset per data (solo le date in comune)
data = df_nikkei.merge(df_sp500, left_index=True, right_index=True, how='inner')

# Pulizia finale (rimuove i NaN risultanti da disallineamenti di giorni non negoziati)
data.dropna(inplace=True)

# Definisci le feature che userai come input (Close è la target e anche feature)
FEATURES = ['Nikkei_Close', 'SP500_Close']
df_features = data[FEATURES]
CLOSE_COLUMN_INDEX = FEATURES.index('Nikkei_Close') 
N_FEATURES = len(FEATURES)


# --- 2. PRE-PROCESSING E NORMALIZZAZIONE ---
# -------------------------------------------
print("2. Pre-processing e Normalizzazione...")

# Normalizzazione di TUTTE le feature
scaler = MinMaxScaler(feature_range=(0, 1))

train_size = int(len(df_features) * TRAIN_SPLIT)
train_data_raw = df_features[:train_size]
test_data_raw = df_features[train_size:] 

# 2. ADDESTRARE LO SCALER SOLO SUI DATI DI TRAINING
scaler.fit(train_data_raw) 

# 3. Trasformare il training set (scaled_data è usato solo per la creazione delle sequenze di train)
train_data_scaled = scaler.transform(train_data_raw) 
# Creazione del set di test scalato, includendo la finestra LOOKBACK dai dati di train
test_data_scaled = scaler.transform(df_features[train_size - LOOKBACK:]) 

# Rinomina i dati scalati per coerenza con il resto del codice:
train_data = train_data_scaled
test_data = test_data_scaled

# Funzione per creare dataset sequenziali multivariati (aggiornata)
def create_multivariate_dataset(dataset, lookback=LOOKBACK, target_index=CLOSE_COLUMN_INDEX):
    X, Y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i, :]) 
        Y.append(dataset[i, target_index])
    return np.array(X), np.array(Y)

X_train, y_train = create_multivariate_dataset(train_data)
X_test, y_test = create_multivariate_dataset(test_data)

print(f"Shape X_train multivariata: {X_train.shape} (Campioni, Lookback, Feature)")


# --- 3. CREAZIONE E ADDESTRAMENTO DEL MODELLO STACKED LSTM ---
# -------------------------------------------------------------
print("3. Creazione e Addestramento del Modello...")

# 3.1. Callback per l'ottimizzazione dell'addestramento
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=30, 
    min_delta=0.00001, 
    verbose=1,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,         
    patience=8,         
    min_lr=0.00001,     
    verbose=1
)

# 3.2. Definizione del Modello
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(LOOKBACK, N_FEATURES))) # Aumentato units a 64
model.add(Dropout(0.3)) 
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(units=1))

# 3.3. Compilazione
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error') # LR ribassato per stabilità

# 3.4. Addestramento del modello
history = model.fit(
    X_train, y_train, 
    epochs=250,          
    batch_size=32,      
    validation_split=0.1, 
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# --- 4. VALUTAZIONE E INVERSIONE DELLA SCALATURA (Test Set) ---
# -------------------------------------------------------------
print("\n4. Valutazione sul Test Set...")

predictions_scaled = model.predict(X_test)

# Inversione della scalatura per ottenere i prezzi reali (solo per la colonna Nikkei_Close)
dummy_array_predictions = np.zeros((len(predictions_scaled), N_FEATURES))
dummy_array_predictions[:, CLOSE_COLUMN_INDEX] = predictions_scaled[:, 0]
predictions = scaler.inverse_transform(dummy_array_predictions)[:, CLOSE_COLUMN_INDEX]

dummy_array_real = np.zeros((len(y_test), N_FEATURES))
dummy_array_real[:, CLOSE_COLUMN_INDEX] = y_test.reshape(-1)
real_prices = scaler.inverse_transform(dummy_array_real)[:, CLOSE_COLUMN_INDEX]

# Calcolo dell'errore (RMSE)
rmse = np.sqrt(np.mean((predictions - real_prices)**2))
print(f"\nRoot Mean Squared Error (RMSE) sul Test Set: {rmse:.2f} Yen")

# Creazione di un DataFrame per il plot
train_length = len(data) - len(real_prices)
df_plot = data[['Nikkei_Close']].iloc[train_length:].rename(columns={'Nikkei_Close': 'Close'})
df_plot['Previsioni'] = predictions
df_plot['Reale'] = real_prices


# --- 5. PREVISIONE FUTURA A 1 MESE (RICORSIVA) ---
# --------------------------------------------------
print("\n5. Generazione Previsioni Ricorsive...")

last_lookback_sequence = scaled_data[-LOOKBACK:].reshape(1, LOOKBACK, N_FEATURES)
future_predictions_scaled = []
current_input = last_lookback_sequence

# Loop Ricorsivo per 20 giorni di trading futuri
for i in range(FUTURE_DAYS):
    predicted_day_scaled = model.predict(current_input, verbose=0)[0]
    future_predictions_scaled.append(predicted_day_scaled[0])
    
    new_sequence = current_input[0, 1:, :]
    
   #ultima riga sp500
    last_known_features = new_sequence[-1].copy() 
    last_known_features[CLOSE_COLUMN_INDEX] = predicted_day_scaled[0] # Aggiorna la previsione Nikkei
    
    new_sequence = np.append(new_sequence, [last_known_features], axis=0)
    current_input = new_sequence.reshape(1, LOOKBACK, N_FEATURES)


# Inversione della Scalatura per ottenere i prezzi reali
dummy_array_future = np.zeros((len(future_predictions_scaled), N_FEATURES))
dummy_array_future[:, CLOSE_COLUMN_INDEX] = future_predictions_scaled
future_predictions = scaler.inverse_transform(dummy_array_future)[:, CLOSE_COLUMN_INDEX]

# Creazione delle date future (utilizzando la frequenza di Business Day)
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=FUTURE_DAYS + 1, freq='B')[1:]

# --- 6. VISUALIZZAZIONE DEI RISULTATI ---
# ----------------------------------------
print("6. Visualizzazione dei Risultati...")

plt.figure(figsize=(14, 7))
plt.plot(data['Nikkei_Close'], label='Prezzo Reale Storico', color='blue', alpha=0.5)
plt.plot(df_plot.index, df_plot['Reale'], label='Prezzo Reale Test', color='red')
plt.plot(df_plot.index, df_plot['Previsioni'], label='Previsioni Stacked LSTM (Test)', color='green', linestyle='--')
plt.plot(future_dates, future_predictions, label=f'Forecast a {FUTURE_DAYS} Business Days', color='darkorange', linewidth=2)
plt.title(f'Previsione Nikkei 225 ({TICKER_NIKKEI}) con S&P 500 Esogeno')
plt.xlabel('Data')
plt.ylabel('Prezzo di Chiusura (Yen)')
plt.legend()
plt.grid(True)
plt.show()

print("\nProcesso completato. Base stabile con feature esogena implementata.")


# In[ ]:




