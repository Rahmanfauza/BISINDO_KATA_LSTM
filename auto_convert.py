import os
print("[-] Mem-patch kompabilitas versi NumPy...")
import numpy as np

# Trik Sakti (Monkey-Patch): 
# Ini untuk mengakali masalah "module 'numpy' has no attribute 'object'"
# yang ditimbulkan oleh versi NumPy terbaru yang tidak lagi support 'np.object' bawaan versi lama tensorflowjs
np.object = np.object_
np.bool = np.bool_
np.int = int
np.float = float
np.typeDict = np.sctypeDict

print("[-] Mengimpor TensorFlowJS...")
import tensorflowjs as tfjs
from tensorflow.keras.models import load_model

model_path = 'gesture_model.h5'
out_dir = 'tfjs_model'

print(f"[*] Memuat model lama '{model_path}'...")
model = load_model(model_path)

print(f"[*] Sedang melakukan konversi (Compile ke folder '{out_dir}')...")
# Mengeksekusi konversi secara internal via Python tanpa perlu Command Prompt (CLI)
tfjs.converters.save_keras_model(model, out_dir)

print(f"\n[✔] HORE! Model berhasil diconvert secara mandiri oleh script.")
print(f"[✔] File web Anda (.json dan .bin) sudah siap di dalam folder '{out_dir}'.")
