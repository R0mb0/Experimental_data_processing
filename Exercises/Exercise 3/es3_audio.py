"""
Esercizio 3 - Time Series Analysis: pulizia spettrale di un brano audio.

Il disturbo (diagnosticato con FFT/PSD/spettrogramma) e' un rumore a bassa
frequenza presente per tutta la durata del brano. Una prima stima (cutoff
180 Hz, ordine 4) non si e' rivelata sufficiente: verificando l'attenuazione
reale in dB, il disturbo si estende fino a ~250-300 Hz e domina la musica
vera di 1-2 ordini di grandezza in quella banda. Servono un cutoff piu' alto
e un filtro piu' ripido (ordine maggiore) per sopprimerlo davvero.

Strategia: filtro passa-alto Butterworth (progettato a mano via cascata di
biquad), applicato in modo zero-phase (forward-backward) per non distorcere
la fase dell'audio.
"""

import wave
import numpy as np

FS_EXPECTED = 44100
CUTOFF_HZ = 250.0   # rivisto dopo verifica: 180 Hz non bastava (vedi sopra)
ORDER = 10           # Butterworth ordine 10 = 5 biquad in cascata


# =====================================================================
# 1. Lettura/scrittura WAV (libreria standard, niente scipy.io.wavfile)
# =====================================================================
def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        fs = w.getframerate()
        ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(n)
    assert sampwidth == 2, "atteso PCM 16 bit"
    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float64)
    return data, fs


def write_wav(path, data, fs):
    data_int16 = np.clip(np.round(data), -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data_int16.tobytes())


data, fs = read_wav("Mozart_int16.wav")
print(f"fs = {fs} Hz, canali = {data.shape[1]}, durata = {len(data)/fs:.2f} s")


# =====================================================================
# 2. Filtro passa-alto Butterworth, a mano, come cascata di biquad
# =====================================================================
def butterworth_highpass_biquads(cutoff_hz, fs, order):
    """
    Un Butterworth di ordine N si costruisce come cascata di N/2 sezioni
    biquad del secondo ordine, ciascuna con lo stesso cutoff ma un fattore
    di merito Q diverso, dato dagli angoli dei poli del prototipo analogico:
        theta_k = pi*(2k-1)/(2N),  Q_k = 1/(2*cos(theta_k)),  k=1..N/2
    Ogni biquad e' poi discretizzato con le formule standard "RBJ Audio EQ
    Cookbook" per un filtro passa-alto del secondo ordine.
    """
    assert order % 2 == 0, "per semplicita' consideriamo solo ordini pari"
    w0 = 2 * np.pi * cutoff_hz / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)

    biquads = []
    for k in range(1, order // 2 + 1):
        theta_k = np.pi * (2 * k - 1) / (2 * order)
        Q = 1.0 / (2 * np.cos(theta_k))
        alpha = sin_w0 / (2 * Q)

        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha

        # normalizzati per a0 = 1
        biquads.append((b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0))
    return biquads


def apply_biquad(x, coeffs):
    """Filtro IIR diretto forma II trasposta: y[n] = b0 x[n] + b1 x[n-1]
    + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]."""
    b0, b1, b2, a1, a2 = coeffs
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n in range(len(x)):
        xn = x[n]
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn
    return y


def highpass_filter(x, cutoff_hz, fs, order, zero_phase=True):
    biquads = butterworth_highpass_biquads(cutoff_hz, fs, order)
    y = x.copy()
    for coeffs in biquads:
        y = apply_biquad(y, coeffs)
    if zero_phase:
        # filtfilt "fatto in casa": applica di nuovo il filtro al segnale
        # invertito nel tempo, poi reinverte il risultato. Elimina lo
        # sfasamento (fase non lineare) tipico di un IIR causale, al
        # prezzo di raddoppiare l'ordine effettivo della risposta in
        # ampiezza (qui equivalente a un Butterworth di ordine 2*order).
        y = y[::-1]
        for coeffs in biquads:
            y = apply_biquad(y, coeffs)
        y = y[::-1]
    return y


# --- Applica il filtro a ciascun canale ---
print(f"\nFiltro Butterworth passa-alto: cutoff={CUTOFF_HZ} Hz, ordine={ORDER}, zero-phase")
cleaned = np.zeros_like(data)
for c in range(data.shape[1]):
    cleaned[:, c] = highpass_filter(data[:, c], CUTOFF_HZ, fs, ORDER)

write_wav("Mozart_cleaned.wav", cleaned, fs)
print("Salvato Mozart_cleaned.wav")


# =====================================================================
# 3. PSD (periodogramma) e spettrogramma (STFT), prima e dopo
# =====================================================================
def psd(x, fs):
    """Periodogramma con finestra di Hanning (riduce il leakage spettrale)."""
    N = len(x)
    spec = np.fft.rfft(x * np.hanning(N))
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    return freqs, np.abs(spec) ** 2


def spectrogram(x, fs, win_len=2048, hop=512):
    """STFT: FFT su finestre sovrapposte -> mappa tempo-frequenza."""
    window = np.hanning(win_len)
    n_frames = 1 + (len(x) - win_len) // hop
    mat = np.zeros((win_len // 2 + 1, n_frames))
    for i in range(n_frames):
        seg = x[i * hop : i * hop + win_len] * window
        mat[:, i] = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(win_len, d=1 / fs)
    times = np.arange(n_frames) * hop / fs
    return freqs, times, mat


mono_orig = data.mean(axis=1)
mono_clean = cleaned.mean(axis=1)

freqs_psd, Po = psd(mono_orig, fs)
_, Pc = psd(mono_clean, fs)

frac_before = 100 * Po[freqs_psd < CUTOFF_HZ].sum() / Po.sum()
frac_after = 100 * Pc[freqs_psd < CUTOFF_HZ].sum() / Pc.sum()
print(f"\nEnergia sotto {CUTOFF_HZ:.0f} Hz  -  prima: {frac_before:.2f}%   dopo: {frac_after:.2f}%")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    axs[0, 0].semilogy(freqs_psd, Po)
    axs[0, 0].set_xlim(0, 2000)
    axs[0, 0].set_title("PSD originale")
    axs[0, 0].set_xlabel("Freq (Hz)")

    axs[0, 1].semilogy(freqs_psd, Pc)
    axs[0, 1].set_xlim(0, 2000)
    axs[0, 1].set_title(f"PSD pulita (cutoff={CUTOFF_HZ:.0f} Hz, ordine={ORDER})")
    axs[0, 1].set_xlabel("Freq (Hz)")

    f_stft, t_stft, mat_o = spectrogram(mono_orig, fs)
    _, _, mat_c = spectrogram(mono_clean, fs)
    axs[1, 0].pcolormesh(t_stft, f_stft, 20 * np.log10(mat_o + 1e-6), shading="auto")
    axs[1, 0].set_ylim(0, 2000)
    axs[1, 0].set_title("Spettrogramma originale")
    axs[1, 0].set_xlabel("Tempo (s)")
    axs[1, 0].set_ylabel("Freq (Hz)")

    axs[1, 1].pcolormesh(t_stft, f_stft, 20 * np.log10(mat_c + 1e-6), shading="auto")
    axs[1, 1].set_ylim(0, 2000)
    axs[1, 1].set_title("Spettrogramma pulito")
    axs[1, 1].set_xlabel("Tempo (s)")

    plt.tight_layout()
    plt.savefig("mozart_before_after.png", dpi=110)
    print("Salvato mozart_before_after.png")
except ImportError:
    print("matplotlib non disponibile: salto i grafici PSD/spettrogramma")
