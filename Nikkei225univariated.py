#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- 1. CONFIGURAZIONE E RACCOLTA DATI (UNIVARIATA) ---
# -----------------------------------------------------
print("1. Configurazione e Raccolta Dati (Nikkei Univariato)...")

# Variabili di configurazione
TICKER_NIKKEI = "^N225"
START_DATE = "2015-01-01"

# Parametri del Modello Ottimizzati e Stabili
LOOKBACK = 120  
TRAIN_SPLIT = 0.8 
FUTURE_DAYS = 20 

# Download dei dati storici


try:
    print(f"Tentativo di download di {TICKER_NIKKEI} tramite yfinance...")
    data = yf.download(TICKER_NIKKEI, start=START_DATE)
except Exception as e:
    print(f"ERRORE di Download: Impossibile scaricare i dati da yfinance. {e}")
    print("Controlla la tua connessione o riprova più tardi.")
# Selezione della singola feature
FEATURES = ['Close']
df_features = data[FEATURES]
N_FEATURES = 1 

# Pulizia finale dei NaN (solo per sicurezza)
df_features.dropna(inplace=True)


# --- 2. PRE-PROCESSING E NORMALIZZAZIONE CORRETTA ---
# ------------------------------------------------------
print("2. Pre-processing e Normalizzazione...")

# Suddivisione in train e test (sui dati NON SCALATI)
train_size = int(len(df_features) * TRAIN_SPLIT)
train_data_raw = df_features[:train_size] 
test_data_raw = df_features[train_size:] 

# ADDESTRARE LO SCALER SOLO SUI DATI DI TRAINING
scaler = MinMaxScaler(feature_range=(0, 1))


train_data_fit = train_data_raw.values.reshape(-1, 1)

# Esegui il fit sulla matrice bidimensionale
scaler.fit(train_data_fit) 

# Trasformare il training set 
train_data_scaled = scaler.transform(train_data_fit)



# 2. Trasformare i dati
train_data_scaled = scaler.transform(train_data_raw) 
# Per il test set, includi la finestra LOOKBACK dai dati di training
test_data_scaled = scaler.transform(df_features[train_size - LOOKBACK:]) 

# Per coerenza:
train_data = train_data_scaled
test_data = test_data_scaled

# Funzione per creare dataset sequenziali univariati (adattata per la forma)
def create_univariate_dataset(dataset, lookback=LOOKBACK):
    X, Y = [], []
    for i in range(lookback, len(dataset)):
        # X: Sequenza di 'lookback' valori (solo 1 feature)
        X.append(dataset[i-lookback:i, 0]) 
        # Y: Valore da predire (il prezzo di chiusura 'Close' del giorno successivo)
        Y.append(dataset[i, 0])
    # Reshape X per l'input LSTM: [samples, lookback, features]
    return np.array(X).reshape(-1, lookback, 1), np.array(Y)

X_train, y_train = create_univariate_dataset(train_data)
X_test, y_test = create_univariate_dataset(test_data)

print(f"Shape X_train univariata: {X_train.shape} (Campioni, Lookback, 1 Feature)")


# --- 3. CREAZIONE E ADDESTRAMENTO DEL MODELLO STACKED LSTM ---
# -------------------------------------------------------------
print("3. Creazione e Addestramento del Modello...")

# 3.1. Callback (Mantenuti per la stabilità)
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=35, 
    min_delta=0.0001, 
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

# 3.2. Definizione del Modello (Complessità ridotta per evitare overfitting)
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(LOOKBACK, N_FEATURES))) 
model.add(Dropout(0.4)) 
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(units=1))

# 3.3. Compilazione
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error') # LR molto basso per stabilità massima

# 3.4. Addestramento del modello
history = model.fit(
    X_train, y_train, 
    epochs=200,          
    batch_size=32,      
    validation_split=0.1, 
    callbacks=[early_stop, reduce_lr], 
    verbose=1
)

# --- 4. VALUTAZIONE E INVERSIONE DELLA SCALATURA (Test Set) ---
# -------------------------------------------------------------
print("\n4. Valutazione sul Test Set...")

predictions_scaled = model.predict(X_test)

# Inversione della scalatura univariata (più semplice)
predictions = scaler.inverse_transform(predictions_scaled)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calcolo dell'errore (RMSE)
rmse = np.sqrt(np.mean((predictions - real_prices)**2))
print(f"\nRoot Mean Squared Error (RMSE) sul Test Set: {rmse.item():.2f} Yen")

# Creazione di un DataFrame per il plot
train_length = len(data) - len(real_prices)
df_plot = data[FEATURES].iloc[train_length:].rename(columns={'Close': 'Reale'})
df_plot['Previsioni'] = predictions
df_plot['Reale'] = real_prices


# --- 5. PREVISIONE FUTURA A 1 MESE (RICORSIVA) ---
# --------------------------------------------------
print("\n5. Generazione Previsioni Ricorsive...")

full_data_array = df_features.values.reshape(-1, 1)


scaled_data_all = scaler.transform(full_data_array) 

# Usa scaled_data_all per prendere l'ultima sequenza
last_lookback_sequence = scaled_data_all[-LOOKBACK:].reshape(1, LOOKBACK, 1) 
future_predictions = []
current_input = last_lookback_sequence

for i in range(FUTURE_DAYS):
    predicted_day_scaled = model.predict(current_input, verbose=0)[0, 0]
    
    
    new_sequence = np.append(current_input[0, 1:, :], [[predicted_day_scaled]], axis=0)
    current_input = new_sequence.reshape(1, LOOKBACK, 1)
    
    
    future_predictions.append(predicted_day_scaled)

# Inversione della Scalatura per ottenere i prezzi reali
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Creazione delle date future
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date, periods=FUTURE_DAYS + 1, freq='B')[1:]

# --- 6. VISUALIZZAZIONE DEI RISULTATI ---
# ----------------------------------------
print("6. Visualizzazione dei Risultati...")

plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Prezzo Reale Storico', color='blue', alpha=0.5)
plt.plot(df_plot.index, df_plot['Reale'], label='Prezzo Reale Test', color='red')
plt.plot(df_plot.index, df_plot['Previsioni'], label='Previsioni Stacked LSTM (Test)', color='green', linestyle='--')
plt.plot(future_dates, future_predictions, label=f'Forecast a {FUTURE_DAYS} Business Days', color='darkorange', linewidth=2)
plt.title(f'Previsione Nikkei 225 ({TICKER_NIKKEI}) Univariata Ottimizzata')
plt.xlabel('Data')
plt.ylabel('Prezzo di Chiusura (Yen)')
plt.legend()
plt.grid(True)
plt.show()

print("\nProcesso completato. Il modello univariato ottimizzato è pronto per l'addestramento.")

