import cv2
import numpy as np
import os
import mediapipe as mp

# Path Penyimpanan Utama
DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

# Parameter Frame
sequence_length = 30 # Frame per gerakan

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    # Pose: 33 landmarks * 4 (x,y,z,visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # LH: 21 * 3 = 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # RH: 21 * 3 = 63
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # 132 + 63 + 63 = 258 fitur (Tanpa Wajah)
    return np.concatenate([pose, lh, rh])

def main():
    print("\n" + "="*60)
    print("🎥 PROGRAM PENGUMPUL DATASET GESTURE DINAMIS")
    print("="*60)
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    # Inisialisasi Kamera 1 Kali, Dipakai Berkali-kali
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True: # Loop Raksasa: Terus diulang sampai pilih N (Selesai)
            
            # Pindai folder yang sudah ada untuk memberi informasi ke User
            existing_actions = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
            if existing_actions:
                print(f"\n📂 Dataset Kata Saat Ini: {existing_actions}")
            
            # 1. Pilih/Masukkan Nama Kata Baru
            action = input("\n[?] Ingin *BUAT* kata baru atau *TAMBAH* kata lama? (Ketikan namanya dsini): ").strip()
            
            if not action:
                print("Nama tidak boleh kosong! Coba lagi.")
                continue
                
            action_path = os.path.join(DATASET_PATH, action)
            os.makedirs(action_path, exist_ok=True) # Buat foldernya jika kata baru
            
            existing_files = len([f for f in os.listdir(action_path) if f.endswith('.npy')])
            print(f"📊 Kata '{action}' di folder Anda saat ini punya total *{existing_files} buah sampel/take*.")
            
            # 2. Input jumlah data tambahan
            try:
                no_sequences = input(f"[?] Berapa jumlah sample/video '{action}' yang direkam sekarang? (Misal: 30) : ")
                no_sequences = int(no_sequences)
            except ValueError:
                print("Format salah! Masukkan berupa Tipe Angka saja.")
                continue
                
            if no_sequences <= 0:
                print("Proses diskip karena 0.")
                continue
            
            # Mendukung auto-resume agar nama file tidak nimpa (Misal mulai dr 30, lalu ke 31, dst)
            start_seq = existing_files
            end_seq = existing_files + no_sequences
            
            print(f"\n[!] BERSIAAAAAAP... Kita akan merekam total {no_sequences} take baru untuk kata: '{action}'...")
            input("[!] Tekan Tombol [ENTER] di terminal ini jika posisi muka&bahu depan kamera sudah SIAP...")
            
            # 3. Looping Perekaman Utama
            try_exit = False
            for sequence in range(start_seq, end_seq):
                if try_exit: break # Lompat jika User Menekan Q
                
                sequence_data = [] # Numpy array penyimpan data berukuran 30 baris
                
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret: 
                        print("ERROR: Sinyal kamera terputus!")
                        break
                        
                    frame = cv2.flip(frame, 1) # Mode cermin
                    
                    # Pemrosesan Holisitik ke Neural Network
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False 
                    results = holistic.process(image_rgb)
                    image_rgb.flags.writeable = True
                    
                    # Render Graphic Visual (No Face)
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                    # ================ Kunci Logika Delay Nicholas YT =================
                    if frame_num == 0: 
                        # Layar Jeda Khusus (Tahan Posisi)
                        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
                        cv2.putText(frame, 'SIAP-SIAP (Tahan Posisi)', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(frame, f'Merekam "{action}" - Take {sequence+1}/{end_seq}', (15,40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.imshow('Kamera Pengumpul Dataset', frame)
                        cv2.waitKey(2000) # Tidur 2 DETIK agar user bisa mereset tangan (Jeda Rekaman)
                    else: 
                        # Layar Take Aktif Berjalan Cepat
                        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
                        cv2.putText(frame, f'MEREKAM! ACTION "{action}" - Take {sequence+1}/{end_seq}', (15,40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.imshow('Kamera Pengumpul Dataset', frame)
                        
                    # Ekstrak data & simpan numpy
                    keypoints = extract_keypoints(results)
                    sequence_data.append(keypoints)
                    
                    # Opsi Paksa Keluar Total Mode Rekaman
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print(f"\n[X] Perekaman digagalkan di tengah jalan pada sample ke-{sequence}.")
                        try_exit = True
                        break
                        
                # Menit ini = Selesai 30 frame, langsung Save!
                if not try_exit:
                    npy_path = os.path.join(action_path, str(sequence))
                    np.save(npy_path, np.array(sequence_data))
                    print(f"   [+] Data kata '{action}' take ke-{sequence+1} SUKSES tersimpan!")
                
            if try_exit:
                break # Keluar dari Looping Raksasa
                
            print(f"\n[✔] Hore! Rekaman kata '{action}' selesai.")
            
            # 4. Melompat ke kata lain? (Interaktif)
            lanjut = input("\n[?] Ingin membuat dataset KATA BARU LAGI sekarang? (y/n) : ").strip().lower()
            if lanjut != 'y':
                print("Selamat istirahat! Skrip akan ditutup.")
                break
                
    # Menghapus Memori Total Hardware 
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
