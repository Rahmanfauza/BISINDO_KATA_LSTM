import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras.models import load_model

# Fungsi Visualisasi Bar Probabilitas (Sama Persis Video YT Nicholas Renotte)
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    # Warna dasar: Oranye, Hijau, Biru, Merah Tua, Ungu (Sesuai jumlah tindakan)
    colors = [(245,117,16), (117,245,16), (16,117,245), (255,50,50), (128,0,128)]
    
    for num, prob in enumerate(res):
        color = colors[num] if num < len(colors) else (200, 200, 200)
        
        # Gambar balok (dikalikan 300 agar terlihat panjang di rasio layar 640x480)
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob * 300), 90+num*40), color, -1)
        
        # Letakkan nama action di bar
        cv2.putText(output_frame, actions[num], (5, 85+num*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # Tulis presentasi akurasinya
        cv2.putText(output_frame, f"{prob*100:.0f}%", (int(prob * 300) + 10, 85+num*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        
    return output_frame

class GestureDetector:
    """
    Detector Backend Action Recognition Sign Language.
    Features: 1662 Dimensions Per Frame.
    """
    def __init__(self, model_path='gesture_model.h5', dataset_path='dataset', threshold=0.6):
        self.threshold = threshold
        self.sequence_length = 30
        self.sequence = []
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.backend = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )
        
        # Load Labels
        self.actions = []
        if os.path.exists(dataset_path):
            for folder in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, folder)):
                    self.actions.append(folder)
        self.actions = np.array(self.actions)
        
        if len(self.actions) == 0:
            raise ValueError("Dataset tidak ditemukan. Harap pastikan folder dataset ada.")
            
        print(f"[*] Memuat model dari '{model_path}'...")
        self.model = load_model(model_path)
        print("[✔] Model berhasil dimuat.")

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.backend.process(image_rgb)
        
        keypoints = self.extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-self.sequence_length:]
        
        probabilities = None
        
        if len(self.sequence) == self.sequence_length:
            # OPTIMASI ANTI DELAY:
            # Pemanggilan `self.model(inputs)` jauh lebih kencang daripada `self.model.predict()` 
            # untuk inferensi realtime di OpenCV. Hal ini menmangkas delay CPU drastis!
            inputs = np.expand_dims(self.sequence, axis=0)
            res = self.model(inputs, training=False).numpy()[0]
            probabilities = res
                
        return results, probabilities

def main():
    try:
        # Threshold cukup 60% agar gesit mendeteksi transisi tangan
        detector = GestureDetector(threshold=0.6) 
    except Exception as e:
        print(f"Error inisialisasi: {e}")
        return

    cap = cv2.VideoCapture(0)
    print("\n[+] Mesin Uji Coba UI Probabilitas Bar (Bebas Delay) Berjalan!")
    print("[+] Tekan 'q' pada layar video untuk keluar.")
    
    sentence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        results, probs = detector.process_frame(frame)
        
        # 1. Gambar Wireframe Tubuh 
        if results.pose_landmarks:
            detector.mp_drawing.draw_landmarks(frame, results.pose_landmarks, detector.mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            detector.mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, detector.mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            detector.mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, detector.mp_holistic.HAND_CONNECTIONS)
        
        # Komen blok face mesh agar tidak nge-lag jika device low-end:
        # if results.face_landmarks:
        #    detector.mp_drawing.draw_landmarks(frame, results.face_landmarks, detector.mp_holistic.FACEMESH_TESSELATION)
            
        # 2. Render Probabilitas AI dan Kata
        if probs is not None:
            # PANGGIL FUNGSI WARNA BAR:
            frame = prob_viz(probs, detector.actions, frame)
            
            class_idx = np.argmax(probs)
            confidence = probs[class_idx]
            
            # LOGIKA RESPONS CEPAT: Langsung perbarui kalimat tanpa nunggu 10 frame stabil!
            if confidence > detector.threshold:
                if len(sentence) > 0:
                    # Mencegah penulisan kata "hallo" berkali-kali ("hallo hallo hallo") 
                    if detector.actions[class_idx] != sentence[-1]:
                        sentence.append(detector.actions[class_idx])
                else:
                    sentence.append(detector.actions[class_idx])
            
            # Kalimat panjang dipotong agar tidak memakan layar
            if len(sentence) > 5:
                sentence = sentence[-5:]
                
        # 3. Merender Teks Kalimat di Kotak Atas
        cv2.rectangle(frame, (0, 0), (640, 40), (220, 220, 220), -1)
        cv2.putText(frame, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('Deteksi Isyarat LSTM - Nicholas Style', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
