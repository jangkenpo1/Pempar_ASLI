from mpi4py import MPI
import pandas as pd
import numpy as np
import time 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

X_latih = None
y_latih = None
X_uji   = None
y_uji   = None

if rank == 0:     
    # 1. Tambahkan parameter decimal=',' agar Pandas tahu koma adalah desimal
    data_latih = pd.read_csv('star_classification_Latih.csv', sep=';', decimal=',') 
    data_uji   = pd.read_csv('star_classification_Uji.csv',   sep=';', decimal=',') 

    fitur = ['delta', 'u', 'g', 'r', 'i', 'z', 'redshift']

    # 2. PENGAMAN TAMBAHAN: Ubah paksa semua fitur menjadi angka (float)
    # Jika masih ada koma yang nyasar sebagai teks, kita replace dengan titik lewat kode ini
    for kolom in fitur:
        if data_latih[kolom].dtype == 'object':
            data_latih[kolom] = data_latih[kolom].astype(str).str.replace(',', '.').astype(float)
        if data_uji[kolom].dtype == 'object':
            data_uji[kolom] = data_uji[kolom].astype(str).str.replace(',', '.').astype(float)

    # 3. Baru ambil nilainya (sekarang dijamin 100% angka)
    X_latih_raw = data_latih[fitur].values 
    y_latih     = data_latih['class'].values 

    # Lanjutkan normalisasi Min-Max seperti biasa ...
    X_min = X_latih_raw.min(axis=0) 
    X_max = X_latih_raw.max(axis=0) 
    X_latih = (X_latih_raw - X_min) / (X_max - X_min) 

    X_uji_raw  = data_uji[fitur].values 
    X_uji_full = (X_uji_raw - X_min) / (X_max - X_min) 
    y_uji_full = data_uji['class'].values 

    # Memecah data menggunakan np.array_split (jauh lebih efisien)
    X_uji = np.array_split(X_uji_full, size) 
    y_uji = np.array_split(y_uji_full, size) 

# 2. BROADCAST & SCATTER
X_latih = comm.bcast(X_latih, root=0)
y_latih = comm.bcast(y_latih, root=0)

X_uji_lokal = comm.scatter(X_uji, root=0)
y_uji_lokal = comm.scatter(y_uji, root=0)


def knn_klasifikasi(X_latih, y_latih, titik_baru, k):
    # Hitung jarak Euclidean menggunakan kalkulasi vektor (sangat cepat dibanding perulangan For)
    jarak = np.sqrt(np.sum((X_latih - titik_baru) ** 2, axis=1))
    
    # Ambil index dari K tetangga terdekat (argpartition jauh lebih ringan dari penyortiran penuh)
    # Pastikan k tetangga benar-benar k yang paling dekat
    idx_k_kasar = np.argpartition(jarak, k)[:k]         # kandidat k terkecil
    idx_k = idx_k_kasar[np.argsort(jarak[idx_k_kasar])] # urutkan ulang yang k itu
    labels_k = y_latih[idx_k]
    
    # Dapatkan label kandidat bedasarkan index
    labels_k = y_latih[idx_k]
    
    # Menghitung suara dan menentukan label tebakan (Prediksi Final)
    unik, jumlah = np.unique(labels_k, return_counts=True)
    prediksi_final = unik[np.argmax(jumlah)]
            
    return prediksi_final

# 4. EKSEKUSI PARALEL
K = 10  
benar_lokal = 0
jumlah_data_lokal = len(X_uji_lokal)

comm.Barrier()
start_time = time.time()

for i in range(jumlah_data_lokal):
    titik_baru = X_uji_lokal[i]
    label_asli = y_uji_lokal[i]
    
    tebakan = knn_klasifikasi(X_latih, y_latih, titik_baru, K)
    
    if tebakan == label_asli:
        benar_lokal += 1

# 5. GATHER
total_benar = comm.reduce(benar_lokal, op=MPI.SUM, root=0)
total_diuji = comm.reduce(jumlah_data_lokal, op=MPI.SUM, root=0)

if rank == 0:
    end_time = time.time()
    akurasi = (total_benar / total_diuji) * 100
    waktu_eksekusi = end_time - start_time
    
    print("\nHASIL EVALUASI KNN PARALEL :")
    print(f"Nilai K             : {K}")
    print(f"Core (N)            : {size}")
    print(f"Total Data Latih    : {len(X_latih)}")
    print(f"Total Data Diuji    : {total_diuji}")
    print(f"Total Tebakan Benar : {total_benar}")
    print(f"Akurasi             : {akurasi:.2f}%")
    print(f"Waktu Eksekusi      : {waktu_eksekusi:.2f} detik")
