"""
Esercizio 3 - Time Series Analysis: pulizia spettrale di un brano audio.
Versione con scipy: usa esattamente i pacchetti indicati dal Readme
(scipy.io.wavfile, numpy, scipy.signal, scipy.fft, matplotlib.pyplot).

Stessa diagnosi e stessa scelta finale della versione "a mano"
(es3_audio.py): il disturbo copre 0-250 Hz circa, serve un passa-alto
Butterworth con cutoff 250 Hz, ordine 10, applicato zero-phase.

Differenze rispetto alla versione manuale:
- lettura/scrittura WAV con scipy.io.wavfile invece del modulo wave
- filtro progettato con scipy.signal.butter(..., output="sos") invece della
  cascata di biquad scritta a mano (sos = "second-order sections", la stessa
  idea di stabilita' numerica per cui avevamo cascata i biquad a mano)
- filtraggio zero-phase con scipy.signal.sosfiltfilt invece del
  forward-backward fatto in casa
- PSD con scipy.signal.welch (media su piu' segmenti sovrapposti, riduce
  la varianza della stima rispetto a un singolo periodogramma) invece di
  un singolo periodogramma con rfft
- spettrogramma con scipy.signal.spectrogram invece della STFT manuale
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, welch, spectrogram

CUTOFF_HZ = 250.0
ORDER = 10


# --- 1. Lettura WAV ---
fs, data = wavfile.read("Mozart_int16.wav")
data = data.astype(np.float64)
print(f"fs = {fs} Hz, canali = {data.shape[1]}, durata = {len(data)/fs:.2f} s")


# --- 2. Filtro Butterworth passa-alto (sos = second-order sections) ---
# "sos" e' la rappresentazione a cascata di sezioni del secondo ordine:
# stesso motivo per cui, a mano, avevamo costruito una cascata di biquad
# invece di un unico filtro di ordine alto in forma diretta (instabilita'
# numerica dei coefficienti per ordini alti in forma diretta).
sos = butter(ORDER, CUTOFF_HZ, btype="highpass", fs=fs, output="sos")

# --- 3. Filtraggio zero-phase (equivalente a scipy.signal.filtfilt,
#         ma che lavora sulla rappresentazione sos, piu' stabile per
#         ordini alti) ---
cleaned = np.zeros_like(data)
for c in range(data.shape[1]):
    cleaned[:, c] = sosfiltfilt(sos, data[:, c])

cleaned_int16 = np.clip(np.round(cleaned), -32768, 32767).astype(np.int16)
wavfile.write("Mozart_cleaned_scipy.wav", fs, cleaned_int16)
print("Salvato Mozart_cleaned_scipy.wav")


# --- 4. PSD con il metodo di Welch (prima/dopo) ---
# Welch: divide il segnale in segmenti sovrapposti, calcola il periodogramma
# di ciascuno e li media. Riduce la varianza della stima rispetto a un
# singolo periodogramma su tutto il segnale (al costo di una risoluzione
# in frequenza leggermente minore, legata alla lunghezza del segmento).
mono_orig = data.mean(axis=1)
mono_clean = cleaned.mean(axis=1)

f_psd, Po = welch(mono_orig, fs=fs, window="hann", nperseg=4096)
_, Pc = welch(mono_clean, fs=fs, window="hann", nperseg=4096)

frac_before = 100 * Po[f_psd < CUTOFF_HZ].sum() / Po.sum()
frac_after = 100 * Pc[f_psd < CUTOFF_HZ].sum() / Pc.sum()
print(f"\nEnergia sotto {CUTOFF_HZ:.0f} Hz  -  prima: {frac_before:.2f}%   dopo: {frac_after:.2f}%")


# --- 5. Spettrogramma (prima/dopo) ---
f_stft, t_stft, Sxx_o = spectrogram(mono_orig, fs=fs, window="hann", nperseg=2048, noverlap=2048 - 512)
_, _, Sxx_c = spectrogram(mono_clean, fs=fs, window="hann", nperseg=2048, noverlap=2048 - 512)


# --- 6. Grafici ---
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs[0, 0].semilogy(f_psd, Po)
    axs[0, 0].set_xlim(0, 2000)
    axs[0, 0].set_title("PSD originale (Welch)")
    axs[0, 0].set_xlabel("Freq (Hz)")

    axs[0, 1].semilogy(f_psd, Pc)
    axs[0, 1].set_xlim(0, 2000)
    axs[0, 1].set_title(f"PSD pulita (cutoff={CUTOFF_HZ:.0f} Hz, ordine={ORDER})")
    axs[0, 1].set_xlabel("Freq (Hz)")

    axs[1, 0].pcolormesh(t_stft, f_stft, 10 * np.log10(Sxx_o + 1e-12), shading="auto")
    axs[1, 0].set_ylim(0, 2000)
    axs[1, 0].set_title("Spettrogramma originale")
    axs[1, 0].set_xlabel("Tempo (s)")
    axs[1, 0].set_ylabel("Freq (Hz)")

    axs[1, 1].pcolormesh(t_stft, f_stft, 10 * np.log10(Sxx_c + 1e-12), shading="auto")
    axs[1, 1].set_ylim(0, 2000)
    axs[1, 1].set_title("Spettrogramma pulito")
    axs[1, 1].set_xlabel("Tempo (s)")

    plt.tight_layout()
    plt.savefig("mozart_before_after_scipy.png", dpi=110)
    print("Salvato mozart_before_after_scipy.png")
except ImportError:
    print("matplotlib non disponibile: salto i grafici")


# --- 7. Verifica della risposta in frequenza del filtro (opzionale) ---
from scipy.signal import sosfreqz

w, h = sosfreqz(sos, worN=8000, fs=fs)
i_cut = np.argmin(np.abs(w - CUTOFF_HZ))
print(f"\n|H(f={CUTOFF_HZ:.0f} Hz)| = {20*np.log10(np.abs(h[i_cut])):.3f} dB  (atteso: -3.01 dB)")
