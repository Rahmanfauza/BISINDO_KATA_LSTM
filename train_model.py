import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

DATASET_PATH = 'dataset'

actions = []
if os.path.exists(DATASET_PATH):
    for folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
            if len(files) > 0:
                actions.append(folder)

actions = np.array(actions)
if len(actions) == 0:
    print("Dataset kosong. Jalankan collect_data.py dulu.")
    exit()

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATASET_PATH, action)
    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    for file in files:
        res = np.load(os.path.join(action_path, file))
        # Validasi Data Corrupt (Harus persis 258)
        if res.shape == (30, 258):
            sequences.append(res)
            labels.append(label_map[action])

X = np.array(sequences)

if len(X) == 0:
    print("[ERROR] Data korup atau belum diupdate ke versi baru (258 fitur).")
    print("Tolong HAPUS folder 'dataset' yang lama secara manual, dan jalankan collect_data.py lagi!")
    exit()
    
# One-hot encoding labels bergantung pada total Class    
y = to_categorical(labels, num_classes=len(actions)).astype(int)

# Split Dataset (Train 90%, Test 10%)
if len(actions) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
else:
    print("[PERINGATAN] Anda merekam kurang dari 2 variasi gaya gesture. Lakukan recording `idle` agar hasil normal!")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Set Logging Folder (agar bisa melacak loss curve nanti)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build LSTM Neural Network matching YT Tutorial
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# ... (Pastikan blok di atasnya tetap seperti sebelumnya, kita hanya ubah model)
model = Sequential()
# Ganti aktivasi 'relu' dengan aktivasi bawaan LSTM (tanh) yang jauh lebih stabil
# karena 'relu' pada LSTM sering menyebabkan exploding gradient (Loss membengkak)
model.add(LSTM(64, return_sequences=True, input_shape=(30, 258))) # aktivasi default: tanh
model.add(Dropout(0.2)) # Menambah dropout agar tidak overfit
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Atur learning rate sedikit lebih kecil 
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0005)

# Compile
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Fit Model dengan Early Stopping untuk otomatis stop jika akurasi sudah bagus
early_stop = EarlyStopping(monitor='categorical_accuracy', patience=30, restore_best_weights=True)

print("\n--- Memulai Training Pipeline Khusus (Stable / Anti-Stuck) ---")
# Tangkap jejak sejarah training ke variabel 'history'
history = model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback, early_stop], validation_data=(X_test, y_test))

# Menyimpan
model_name = 'gesture_model.h5'
model.save(model_name)
print(f"\n[✔] Model Holisitic disimpan sebagai '{model_name}'.")

# -----------------
# EVALUASI & VISUALISASI HASIL TRAINING
# -----------------
print("\n--- Membuat Diagram Evaluasi Hasil (Tunggu sebentar...) ---")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

out_path = 'Hasil_Evaluasi'
os.makedirs(out_path, exist_ok=True)

# 1. Diagram Akurasi (Accuracy)
plt.figure(figsize=(10, 5))
plt.plot(history.history['categorical_accuracy'], label='Akurasi Training (Belajar)', color='blue')
plt.plot(history.history['val_categorical_accuracy'], label='Akurasi Validasi (Ujian)', color='orange')
plt.title('Grafik Perkembangan Akurasi AI (Semakin tinggi semakin bagus)')
plt.xlabel('Putaran (Epochs)')
plt.ylabel('Tingkat Akurasi')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_path, 'grafik_akurasi.png'))
plt.close()

# 2. Diagram Loss (Error)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Loss Training', color='red')
plt.plot(history.history['val_loss'], label='Loss Validasi', color='purple')
plt.title('Grafik Tingkat Error AI (Semakin turun/kecil semakin bagus)')
plt.xlabel('Putaran (Epochs)')
plt.ylabel('Nilai Error (Loss)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(out_path, 'grafik_loss.png'))
plt.close()

# 3. Klasifikasi Data Uji (Test Data)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Laporan Evaluasi Penuh
print("\n--- LAPORAN KLASIFIKASI AKURASI AKHIR ---")
# Mencegah error jika label tidak terekam di y_test
report = classification_report(y_true_classes, y_pred_classes, target_names=actions, zero_division=0)
print(report)

# Simpan text report
with open(os.path.join(out_path, 'laporan_metrik.txt'), 'w') as f:
    f.write(report)

# 4. Diagram Confusion Matrix (Melihat dimana AI sering 'Terkecoh')
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=actions, yticklabels=actions)
plt.title('Confusion Matrix (Distribusi Ketepatan Tebak)')
plt.xlabel('Tebakan AI (Prediksi)')
plt.ylabel('Status Asli di Dunia Nyata (Fakta Actual)')
plt.savefig(os.path.join(out_path, 'confusion_matrix.png'))
plt.close()

print(f"\n[✔] SUCCESS! Semua diagram grafik, matriks ketepatan, dan laporan evaluasi telah di-generate otomatis ke dalam folder '{out_path}'!")
