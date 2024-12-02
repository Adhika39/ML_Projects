import pandas as pd
import numpy as np
import seaborn as sns
import string
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('healthcare_dataset.csv')

df = df.fillna(df.mean(numeric_only=True))

categorical_columns = ['Insurance Provider', 'Hospital', 'Admission Type', 'Medication', 'Test Results', 'Medical Condition']

label_encoder = LabelEncoder()
for col in categorical_columns:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

df['Date of Admission'] = pd.to_datetime(df['Date of Admission'], errors='coerce')
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'], errors='coerce')
df['Stay Duration'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
df = df.drop(columns=['Date of Admission', 'Discharge Date'])
print(df)

numerical_columns = ['Age', 'Billing Amount', 'Stay Duration']
scaler = MinMaxScaler(feature_range=(0, 1))
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print(df)

target_column = 'Billing Amount'


non_numeric_columns = ['Name', 'Doctor', 'Gender','Blood Type']
df_numeric = df.drop(columns=non_numeric_columns)


features = df_numeric.drop(columns=[target_column]).values.astype(np.float32)
y = df[target_column].values.astype(np.float32)

sequence_length = 20
X = np.array([features[i:i + sequence_length] for i in range(len(features) - sequence_length)], dtype=np.float32)
y = y[sequence_length:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

optimizer = Adam(learning_rate=0.0005)
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Flatten(),
    Dense(1)  # For regression
])
model.compile(optimizer=optimizer, loss='mse')
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

test_loss = model.evaluate(X_test, y_test)
print("Test Loss (MSE):", test_loss)

pred = np.argmax(model.predict(X_test[7:8]))
print("pred ", pred)
print("y_test[7:8] ", y_test[7:8])

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss during Training')
plt.show()
