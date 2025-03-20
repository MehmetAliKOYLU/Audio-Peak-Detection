import moviepy.editor as mp
import numpy as np
import librosa
import librosa.display
import scipy.signal as signal
import matplotlib.pyplot as plt

# Video dosyasının yolu
video_path = "Etüt.mp4"
audio_path = "Etüt.wav"

# Videodan sesi çıkarıp WAV formatında kaydetme
video = mp.VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

# WAV dosyasını yükleme
y, sr = librosa.load(audio_path, sr=None)

# Gürültüyü azaltmak için yüksek geçiren filtre uygulama
b, a = signal.butter(4, 1000/(sr/2), btype='high')
filtered_y = signal.filtfilt(b, a, y)

# Enerji hesaplama
frame_length = 1024
hop_length = 512  
energy = np.array([sum(abs(filtered_y[i:i+frame_length]**2)) for i in range(0, len(filtered_y), hop_length)])

# Eşik değeri belirleme
threshold = np.percentile(energy, 99)

# Peak noktalarını bulma

all_peaks, _ = signal.find_peaks(energy, height=threshold*3.10, distance=frame_length/10)

# Eşik değerinin altındaki noktaları filtreleme
peaks = [p for p in all_peaks if energy[p] > threshold]

#Liste NumPy dizisi
peaks = np.array(peaks, dtype=int)


print(f"BULUNAN PEAK NOKTALARI: {len(peaks)}")
print(f"PEAK İNDEKSLERİ: {peaks}")
print(f"PEAK ENERJİLERİ: {energy[peaks]}")


fig, axes = plt.subplots(3, 1, figsize=(12, 9))

#Orijinal ses dalgası
axes[0].plot(y, alpha=0.5, label="Orijinal Ses Dalgası", color="skyblue")
axes[0].set_title("Orijinal Ses Dalgası")
axes[0].legend()

# Filtrelenmiş ses dalgası
axes[1].plot(filtered_y, alpha=0.5, label="Filtrelenmiş Ses Dalgası", color="orange")
axes[1].set_title("Filtrelenmiş Ses Dalgası")
axes[1].legend()

# Enerji grafiği ve vuruş tespitleri
axes[2].plot(energy, label="Enerji", color='blue')
axes[2].scatter(peaks, energy[peaks], color='red', s=50, edgecolors='black', label="Tespit Edilen Vuruşlar")
axes[2].axhline(y=threshold*3.10, color='r', linestyle='--', label="Eşik Değeri")
axes[2].set_title("Enerji Grafiği ve Vuruş Tespiti")
axes[2].legend()

plt.tight_layout()

# Grafikleri kaydetme
updated_energy_graph_path = "enerji_grafigi.png"
fig.savefig(updated_energy_graph_path, dpi=100)

# Güncellenmiş grafiği göster
plt.show()




