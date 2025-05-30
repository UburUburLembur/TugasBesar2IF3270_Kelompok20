# Implementasi dan Analisis Model Deep Learning (CNN, Simple RNN, LSTM)

Repository ini berisi implementasi dan analisis berbagai model deep learning yaitu Convolutional Neural Network (CNN), Simple Recurrent Neural Network (Simple RNN), dan Long Short-Term Memory (LSTM). Proyek ini bertujuan untuk memahami arsitektur, proses forward propagation, serta pengaruh berbagai parameter terhadap kinerja masing-masing model.

## Struktur Repository

Berikut adalah struktur direktori utama dalam repository ini:

-   **/data/**: (Opsional) Berisi dataset yang digunakan atau instruksi untuk mendapatkannya. Juga dapat berisi file bobot model (`.weights.h5`) yang telah dilatih sebelumnya.
-   **/doc/**: Berisi laporan proyek dalam format `.pdf` yang mencakup seluruh aspek tugas, mulai dari deskripsi persoalan hingga kesimpulan.
-   **/src/**: Berisi seluruh kode sumber implementasi model dan notebook pengujian.
    -   **/src/cnn/**: Kode sumber untuk model CNN.
    -   **/src/rnn/**: Kode sumber untuk model Simple RNN.
    -   **/src/lstm/**: Kode sumber untuk model LSTM.
    -   `notebook_pengujian_cnn.ipynb`: Notebook untuk eksperimen dan pengujian model CNN.
    -   `notebook_pengujian_rnn.ipynb`: Notebook untuk eksperimen dan pengujian model Simple RNN.
    -   `notebook_pengujian_lstm.ipynb`: Notebook untuk eksperimen dan pengujian model LSTM.
-   `requirements.txt`: Daftar pustaka Python yang dibutuhkan untuk menjalankan proyek.
-   `README.md`: File ini, berisi informasi mengenai repository.

## Deskripsi Singkat Repository

Proyek ini mencakup implementasi dari nol (atau menggunakan TensorFlow/Keras dengan pemahaman mendalam) untuk model-model berikut:
1.  **Convolutional Neural Network (CNN)**: Untuk tugas klasifikasi gambar (atau tugas lain yang relevan).
2.  **Simple Recurrent Neural Network (Simple RNN)**: Untuk tugas pemrosesan sekuensial (misalnya analisis sentimen teks).
3.  **Long Short-Term Memory (LSTM)**: Varian RNN yang lebih canggih untuk menangani dependensi jangka panjang dalam data sekuensial.

Fokus utama adalah pada pemahaman mekanisme forward propagation dan analisis pengaruh hiperparameter seperti jumlah layer, jumlah filter/unit, ukuran filter, jenis pooling, dan jenis layer rekuren (unidirectional/bidirectional).

## Cara Setup dan Run Program

### Prasyarat

Pastikan Anda telah menginstal perangkat lunak berikut:
* Python (disarankan versi 3.8 atau lebih baru)
* pip (Python package installer)
* Git (untuk mengkloning repository)

### Langkah-langkah Setup

1.  **Clone Repository:**
    Buka terminal atau command prompt, lalu jalankan perintah berikut untuk mengkloning repository:
    ```bash
    git clone https://github.com/UburUburLembur/TugasBesar2IF3270_Kelompok20.git
    cd https://github.com/UburUburLembur/TugasBesar2IF3270_Kelompok20.git
    ```
    
2.  **Instal Dependensi:**
    Setelah virtual environment aktif, instal semua pustaka yang dibutuhkan dengan menjalankan:
    ```bash
    pip install -r requirements.txt
    ```

### Menjalankan Program (Notebook Pengujian)

1.  **Persiapkan Data dan Bobot Model:**
    * Jika dataset tidak disertakan dalam repository, unduh dan letakkan di folder `/data/` sesuai instruksi yang mungkin ada di dalam notebook atau laporan.
    * File bobot model (`.h5`) yang digunakan oleh implementasi custom RNN (dan mungkin juga CNN/LSTM jika Anda memuat bobot yang sudah ada) harus tersedia. Letakkan file bobot di direktori yang sesuai (misalnya `/data/` atau subfolder khusus) dan pastikan path ke file bobot tersebut sudah benar di dalam notebook pengujian.

2.  **Jalankan Jupyter Notebook atau Jupyter Lab:**
    Dari direktori root proyek (`TugasBesarIF3270_Kelomok20`), jalankan:
    ```bash
    jupyter lab
    ```
    atau
    ```bash
    jupyter notebook
    ```
    Ini akan membuka antarmuka Jupyter di browser Anda.

3.  **Buka dan Jalankan Notebook Pengujian:**
    * Navigasi ke direktori `/src/`.
    * Buka notebook yang relevan (misalnya, `notebook_pengujian_rnn.ipynb`, `notebook_pengujian_cnn.ipynb`, atau `notebook_pengujian_lstm.ipynb`).
    * Ikuti instruksi di dalam notebook dan jalankan sel-sel kode untuk melakukan pengujian dan eksperimen.

### Cara Menggunakan Modul Forward Propagation (Implementasi Custom)

Modul ini memungkinkan Anda untuk membangun model lapisan demi lapisan, memuat bobot yang sudah ada, dan melakukan forward pass.

#### 1. CNN

#### 2. Simple RNN / Bidirectional RNN

Berikut adalah contoh dasar penggunaan modul Simple RNN dari `src.rnn.model` di dalam notebook pengujian Anda:

```python
# Asumsikan notebook berada di src
import sys
import os

project_root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

import numpy as np
# Impor kelas-kelas yang dibutuhkan dari modul custom RNN Anda
from src.rnn.model import (Model, EmbeddingLayer, SimpleRNNLayer,
                           BidirectionalSimpleRNNLayer, DenseLayer, DropoutLayer)

# --- 1. Inisialisasi Parameter Model (Contoh) ---
vocab_size = 10000  # Sesuaikan dengan dataset Anda
embedding_dim = 128
input_length = 50   # Panjang sekuens input
rnn_units = 64
dense_units = 1     # Untuk klasifikasi biner, atau jumlah kelas untuk multi-kelas
path_to_weights = "data/bobot_model_rnn_anda.h5" # GANTI DENGAN PATH BOBOT ANDA

# --- 2. Membangun Arsitektur Model ---
# Inisialisasi model utama
custom_model_rnn = Model(name="MyCustomRNN")

# Tambahkan layer Embedding
# PENTING: 'name' pada setiap layer harus SESUAI dengan nama layer
# yang tersimpan di file bobot .h5 Anda.
custom_model_rnn.add(EmbeddingLayer(vocab_size=vocab_size, embedding_dim=embedding_dim, name="embedding_layer_nama_di_h5"))

# Tambahkan layer SimpleRNN (atau BidirectionalSimpleRNNLayer)
# custom_model_rnn.add(SimpleRNNLayer(units=rnn_units, return_sequences=True, name="simplernn_layer_nama_di_h5"))
custom_model_rnn.add(BidirectionalSimpleRNNLayer(units=rnn_units, return_sequences=False, name="bidirectional_rnn_layer_nama_di_h5")) # Jika return_sequences=False, output hanya dari timestep terakhir

# Tambahkan layer Dropout jika ada
custom_model_rnn.add(DropoutLayer(rate=0.5, name="dropout_layer_nama_di_h5")) # Nama layer dropout mungkin tidak selalu ada di file bobot

# Tambahkan layer Dense untuk output
custom_model_rnn.add(DenseLayer(units=dense_units, activation='sigmoid', name="dense_layer_nama_di_h5")) # 'sigmoid' untuk biner, 'softmax' untuk multi-kelas

# --- 3. Memuat Bobot Model ---
try:
    custom_model_rnn.load_weights(path_to_weights)
    print(f"Bobot model berhasil dimuat dari {path_to_weights}")
except Exception as e:
    print(f"ERROR saat memuat bobot: {e}")
    print("Pastikan path_to_weights benar dan nama layer di model.add() sesuai dengan di file .h5.")

# --- 4. Menampilkan Ringkasan Model ---
custom_model_rnn.summary()

# --- 5. Persiapan Input Data (Contoh) ---
# Buat data input dummy (ganti dengan data Anda yang sudah di-tokenize dan di-padding)
# Bentuk input: (batch_size, input_length)
dummy_input_data = np.random.randint(0, vocab_size, size=(2, input_length)) # Contoh batch_size = 2

# --- 6. Melakukan Forward Propagation ---
# Pastikan bobot sudah termuat sebelum melakukan forward pass
if all(getattr(layer, '_is_built', True) for layer in custom_model_rnn.layers): # Cek apakah semua layer yang punya '_is_built' sudah True
    try:
        predictions = custom_model_rnn.forward(dummy_input_data)
        print("\n--- Prediksi Model ---")
        print(predictions)
        print(f"Shape prediksi: {predictions.shape}")
    except Exception as e:
        print(f"ERROR saat forward propagation: {e}")
else:
    print("Bobot model belum sepenuhnya dimuat. Forward pass tidak dapat dilakukan.")
```

#### 3. LSTM / Bidirectional LSTM


## Pembagian Tugas Tiap Anggota Kelompok

Berikut adalah pembagian tugas untuk masing-masing anggota kelompok:


* **Muhammad Zakkiy (10122074)**:
    * Implementasi dasar Simple RNN dan penulisan bagian Simple RNN di laporan
    * Pembuatan notebook pengujian untuk Simple RNN
    * Eksperiman model RNN
    * Pengelolaan repository GitHub
    
* **Ghaisan Zaki Pratama (10122078)**:
    * Implementasi dasar CNN dan penulisan bagian CNN di laporan
    * Pembuatan notebook pengujian untuk Simple CNN
    * Eksperimen model CNN
    * Pengelolaan repository GitHub
  
* **Fardhan Indrayesa (12821046)**:
    * Implementasi dasar LSTM dan penulisan bagian LSTM di laporan
    * Pembuatan notebook pengujian untuk Simple LSTM
    * Eksperimen model LSTM
    * Pengelolaan repository GitHub
