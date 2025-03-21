import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from pydub import AudioSegment
import librosa
import librosa.display
import scipy.ndimage
import soundfile as sf

# Функция для проверки аудиомассива
def check_audio_validity(audio, label="Audio"):
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{label} содержит NaN или бесконечные значения!")

# Функция нормализации аудио
def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    return audio / max_val if max_val > 0 else audio

# Функция для загрузки аудиофайла
def load_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)  # Загружаем файл с исходной частотой дискретизации
    audio = np.nan_to_num(audio)  # Удаляем NaN и бесконечные значения
    audio = normalize_audio(audio)  # Нормализация аудио
    check_audio_validity(audio, "Original Audio")  # Проверяем аудио
    return audio, sr

# Функция для сохранения аудиофайла в MP3
def save_audio(file_path, audio, sr):
    sf.write(file_path, audio, sr, format='MP3')
    print(f"Аудиофайл сохранен как {file_path}")

# Функция для отображения спектра сигнала
def plot_audio(audio, sr):
    if len(audio) == 0:
        raise ValueError("Ошибка: Аудиомассив пустой!")
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.title("Аудиосигнал")
    plt.show()

# Функция для создания полосовых фильтров
def bandpass_filter(audio, sr, lowcut, highcut, order=5):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.lfilter(b, a, audio)
    filtered_audio = np.nan_to_num(filtered_audio)  # Удаляем NaN и бесконечные значения
    filtered_audio = np.clip(filtered_audio, -1.0, 1.0)  # Ограничение амплитуды
    check_audio_validity(filtered_audio, "Filtered Audio")  # Проверяем аудио
    return filtered_audio

# Функция графического эквалайзера (3 полосы)
def equalizer(audio, sr, gains):
    low = bandpass_filter(audio, sr, 20, 300) * gains[0]  # Низкие частоты
    mid = bandpass_filter(audio, sr, 300, 3000) * gains[1]  # Средние частоты
    high = bandpass_filter(audio, sr, 3000, 16000) * gains[2]  # Высокие частоты
    equalized_audio = low + mid + high
    equalized_audio = np.nan_to_num(equalized_audio)  # Удаляем NaN и бесконечные значения
    equalized_audio = np.clip(equalized_audio, -1.0, 1.0)  # Ограничение амплитуды
    equalized_audio = normalize_audio(equalized_audio)  # Нормализация аудио
    check_audio_validity(equalized_audio, "Equalized Audio")  # Проверяем аудио
    return equalized_audio

# Оптимизированная функция реверберации (быстрая свертка)
def reverb(audio, sr, decay=0.5, delay=0.02, num_echoes=5):
    delay_samples = int(sr * delay)
    impulse_response = np.zeros(delay_samples * num_echoes)
    for i in range(num_echoes):
        impulse_response[i * delay_samples] = decay ** i
    reverbed_audio = signal.fftconvolve(audio, impulse_response, mode='full')[:len(audio)]
    reverbed_audio = np.clip(reverbed_audio, -1.0, 1.0)  # Ограничение амплитуды
    reverbed_audio = normalize_audio(reverbed_audio)  # Нормализация аудио
    check_audio_validity(reverbed_audio, "Reverbed Audio")  # Проверяем аудио
    return reverbed_audio

# Функция режекторного фильтра (notch-фильтр)
def notch_filter(audio, sr, freq=50, quality=30):
    nyquist = 0.5 * sr
    freq = freq / nyquist
    b, a = signal.iirnotch(freq, quality)
    filtered_audio = signal.filtfilt(b, a, audio)
    filtered_audio = np.nan_to_num(filtered_audio)  # Удаляем NaN и бесконечные значения
    filtered_audio = np.clip(filtered_audio, -1.0, 1.0)  # Ограничение амплитуды
    check_audio_validity(filtered_audio, "Notch Filtered Audio")  # Проверяем аудио
    return filtered_audio

# Функция удаления щелчков и хлопков (медианный фильтр)
def remove_clicks(audio, kernel_size=5):
    filtered_audio = scipy.ndimage.median_filter(audio, size=kernel_size)
    filtered_audio = np.nan_to_num(filtered_audio)  # Удаляем NaN и бесконечные значения
    filtered_audio = np.clip(filtered_audio, -1.0, 1.0)  # Ограничение амплитуды
    check_audio_validity(filtered_audio, "De-clicked Audio")  # Проверяем аудио
    return filtered_audio

if __name__ == '__main__':

    file_path = "Lilu_-_Dozhdis.mp3"
    audio, sr = load_audio(file_path)

    plot_audio(audio, sr)

    # Применение эквалайзера с усилением средних частот
    gains = [0.8, 1.5, 0.9]  # Коэффициенты усиления (низкие, средние, высокие)
    equalized_audio = equalizer(audio, sr, gains)
    plot_audio(equalized_audio, sr)

    save_audio("processed_audio1.mp3", equalized_audio, sr)

    # Применение реверберации
    reverbed_audio = reverb(audio, sr, decay=0.6, delay=0.03, num_echoes=5)
    plot_audio(reverbed_audio, sr)

    save_audio("processed_audio2.mp3", reverbed_audio, sr)

    # Применение режекторного фильтра (Notch) для удаления шума 50 Гц
    notched_audio = notch_filter(audio, sr, freq=50, quality=30)
    plot_audio(notched_audio, sr)

    save_audio("processed_audio3.mp3", notched_audio, sr)

    declicked_audio = remove_clicks(audio, kernel_size=5)
    plot_audio(declicked_audio, sr)

    save_audio("processed_audio4.mp3", declicked_audio, sr)
