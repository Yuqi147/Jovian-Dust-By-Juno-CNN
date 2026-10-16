"""Inspection and visualization helpers for exported Juno/Waves CSV products.

Provides :func:`read_exported_csv` to reload CSV files written by
:class:`PDS3TableReader`, plus plotting utilities (:func:`plot_waveform`,
:func:`compute_spectrogram`, :func:`plot_spectrogram`) and peak normalization
(:func:`normalize_peak`) for dust-impact waveform analysis.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def read_exported_csv(csv_file):
    """Read a CSV file exported by :meth:`PDS3TableReader.export`.

    Parameters
    ----------
    csv_file : str
        Path to the exported CSV file.

    Returns
    -------
    records : list of dict
        Each dict holds ``product``, ``receiver``, ``time``, ``sclk``, ``dt``,
        ``n`` and the ``waveform`` array. The structure matches the internal
        records of :class:`PDS3TableReader`.

    Raises
    ------
    ValueError
        If the CSV header does not match the expected format.
    """
    records = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Base metadata fields.
        base_fields = ["product", "receiver", "time", "sclk", "dt", "n"]

        if header[:6] != base_fields:
            raise ValueError("CSV header does not match expected format")

        # Determine how the waveform is stored.
        if header[6] == "waveform":
            mode = "string"
        elif header[6].startswith("wf_"):
            mode = "flatten"
        else:
            raise ValueError("Unrecognized waveform format in CSV")

        for row in reader:
            r = {
                "product": row[0],
                "receiver": row[1],
                "time": row[2],
                "sclk": float(row[3]),
                "dt": float(row[4]),
                "n": int(row[5]),
            }

            if mode == "string":
                wf = np.array(
                    [float(x) for x in row[6].split()],
                    dtype=np.float32,
                )
            else:  # flatten
                n = r["n"]
                wf_vals = []
                for x in row[6:6 + n]:
                    if x != "":
                        wf_vals.append(float(x))
                wf = np.array(wf_vals, dtype=np.float32)

            r["waveform"] = wf
            records.append(r)

    return records


def plot_waveform(record, use_time_axis=True):
    """Plot a single waveform record.

    Parameters
    ----------
    record : dict
        One element of the records list returned by :func:`read_exported_csv`.
    use_time_axis : bool
        True  -> x axis = time (seconds)
        False -> x axis = sample index
    """
    wf = record["waveform"]
    n = record["n"]
    dt = record["dt"]

    if use_time_axis:
        x = np.arange(n) * dt
        xlabel = "Time (s)"
    else:
        x = np.arange(n)
        xlabel = "Sample index"

    plt.figure(figsize=(10, 4))
    plt.plot(x, wf, lw=0.8)

    plt.xlabel(xlabel)
    plt.ylabel("Amplitude\nE (V/m)")
    plt.title(
        f"{record['receiver']} | {record['time']}"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_spectrogram(record, nperseg=512, noverlap=256):
    """Compute the short-time Fourier transform (spectrogram) of a waveform record.

    Parameters
    ----------
    record : dict
        Record containing a ``waveform`` array and sampling interval ``dt``.
    nperseg : int
        Length of each STFT segment.
    noverlap : int
        Number of points of overlap between segments.

    Returns
    -------
    (f, t, Sxx) : tuple of ndarray
        Frequencies, segment times, and the spectrogram.
    """
    fs = 1.0 / record["dt"]

    f, t, Sxx = spectrogram(
        record["waveform"],
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
    )

    return f, t, Sxx


def plot_spectrogram(record):
    """Plot the log-power spectrogram of a waveform record."""
    f, t, S = compute_spectrogram(record)

    plt.figure(figsize=(7, 4))
    plt.pcolormesh(t, f, np.log10(S), shading="auto")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.title(record["receiver"])
    plt.colorbar(label="log10 Power")
    plt.tight_layout()
    plt.show()


def normalize_peak(record, eps=1e-30):
    """Normalize a waveform so that max(|waveform|) = 1.

    Parameters
    ----------
    record : dict
        Record that contains a ``waveform`` array.
    eps : float
        Peak threshold below which the waveform is left unchanged (prevents
        division by zero).

    Returns
    -------
    new_record : dict
        Shallow copy of the input record with a normalized ``waveform`` and an
        additional ``norm_factor`` key holding the peak value (1.0 if the peak
        is below ``eps``).
    """
    wf = np.asarray(record["waveform"], dtype=float)
    peak = np.max(np.abs(wf))

    new = record.copy()

    if peak < eps:
        new["waveform"] = wf.copy()
        new["norm_factor"] = 1.0
    else:
        new["waveform"] = wf / peak
        new["norm_factor"] = peak

    return new
