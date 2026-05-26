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
Y_uji   = None

if rank == 0:     

    data_latih = pd.read_csv('data_latih.csv', sep=';', nrows=100000) 
    data_uji   = pd.read_csv('data_uji.csv',   sep=';') 

    fitur = ['weekly_self_study_hours', 'attendance_percentage', 'class_participation'] 

    X_latih_raw = data_latih[fitur].values 
    y_latih     = data_latih['grade'].values 

    X_min = X_latih_raw.min(axis=0) 
    X_max = X_latih_raw.max(axis=0) 
    X_latih = (X_latih_raw - X_min) / (X_max - X_min) 

    X_uji_raw  = data_uji[fitur].values 
    X_uji_full = (X_uji_raw - X_min) / (X_max - X_min) 
    y_uji_full = data_uji['grade'].values 

    # Memecah data menggunakan np.array_split (jauh lebih efisien)
    X_uji = np.array_split(X_uji_full, size) 
    Y_uji = np.array_split(y_uji_full, size) 

# 2. BROADCAST & SCATTER
X_latih = comm.bcast(X_latih, root=0)
y_latih = comm.bcast(y_latih, root=0)

X_uji_lokal = comm.scatter(X_uji, root=0)
y_uji_lokal = comm.scatter(Y_uji, root=0)

# 3. FUNGSI KNN
def knn_klasifikasi(X_latih, y_latih, titik_baru, k):
    # Hitung jarak Euclidean menggunakan kalkulasi vektor 
    jarak = np.sqrt(np.sum((X_latih - titik_baru) ** 2, axis=1))
    
    # Ambil index dari K tetangga terdekat 
    idx_k = np.argpartition(jarak, k)[:k]
    
    # Dapatkan label kandidat bedasarkan index
    labels_k = y_latih[idx_k]
    
    # Menghitung suara dan menentukan label tebakan (Prediksi Final)
    unik, jumlah = np.unique(labels_k, return_counts=True)
    prediksi_final = unik[np.argmax(jumlah)]
            
    return prediksi_final

# 4. EKSEKUSI PARALEL
K = 5  
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