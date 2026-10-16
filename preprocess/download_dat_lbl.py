"""Download Juno/Waves burst-mode waveform data (PDS3 .dat/.lbl files) from the PDS archive.

The script crawls the PDS Planetary Plasma Interactions node directory listing for the
Juno Waves Burst Waveform data set (JNO-E_J_SS-WAV-3-CDR-BSTFULL-V2.0), filters files
by mission year and day-of-year (DOY), and downloads the selected products into a local
directory tree that mirrors the remote layout.
"""

import os
import time

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# =================== User configuration ===================

# Remote root directory of the PDS data set.
BASE_HTTP = (
    "https://pds-ppi.igpp.ucla.edu/data/"
    "JNO-E_J_SS-WAV-3-CDR-BSTFULL-V2.0/DATA/WAVES_BURST/"
)

# Local directory where downloaded files are stored.
OUT_ROOT = r"C:\Users\bacid\Pychondownload2018\WAVES"

# Filtering range for the mission year (inclusive).
START_YEAR = 2022
END_YEAR = 2022

# Filtering range for the day-of-year (inclusive); None disables the bound.
START_DOY = None  # int or None
END_DOY = 36  # int or None

# Delay between file requests to avoid PDS rate limiting (seconds).
SLEEP_BETWEEN_FILES = 0.05

# ===========================================================


session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (JUNO WAVES downloader)"
})


def file_in_range(fname):
    """Return True if a WAV_ filename falls within the configured YEAR / DOY range.

    Parameters
    ----------
    fname : str
        File name of the form ``WAV_YYYYDOY_...``.

    Returns
    -------
    bool
        Whether the file satisfies the configured year and DOY bounds.
    """
    if not fname.startswith("WAV_"):
        return False

    try:
        year = int(fname[4:8])
        doy = int(fname[8:11])
    except Exception:
        return False

    if not (START_YEAR <= year <= END_YEAR):
        return False

    if START_DOY is not None and doy < START_DOY:
        return False

    if END_DOY is not None and doy > END_DOY:
        return False

    return True


def list_directory(url):
    """List the subdirectories and files referenced by an Apache-style directory page.

    Parameters
    ----------
    url : str
        URL of the directory listing.

    Returns
    -------
    (dirs, files) : tuple of list of str
        Subdirectory names (trailing slash) and file names found on the page.
    """
    r = session.get(url, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    dirs, files = [], []

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href or href.startswith("?"):
            continue
        if href.endswith("/"):
            dirs.append(href)
        else:
            files.append(href)

    return dirs, files


def download_file(file_url, local_path):
    """Download a single remote file if it does not already exist locally."""
    if os.path.exists(local_path):
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with session.get(file_url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def main():
    """Crawl the PDS directory tree and download all in-range .dat/.lbl products."""
    print("\n=== JUNO WAVES downloader started ===")
    print(f"YEAR RANGE : {START_YEAR} -> {END_YEAR}")
    print(f"DOY RANGE  : {START_DOY} -> {END_DOY}")
    print("=======================================\n")

    orbit_dirs, _ = list_directory(BASE_HTTP)

    for orbit in orbit_dirs:
        orbit_url = urljoin(BASE_HTTP, orbit)

        print(f"\nEntering directory: {orbit}")

        try:
            _, files = list_directory(orbit_url)
        except Exception as e:
            print(f"  [SKIP] failed to access directory: {e}")
            continue

        for fname in files:
            if not fname.lower().endswith((".dat", ".lbl")):
                continue

            if not file_in_range(fname):
                continue

            file_url = urljoin(orbit_url, fname)

            local_path = os.path.join(
                OUT_ROOT,
                orbit.rstrip("/"),
                fname
            )

            try:
                download_file(file_url, local_path)
                print(f"  [OK] downloaded: {orbit}{fname}")
                time.sleep(SLEEP_BETWEEN_FILES)
            except Exception as e:
                print(f"  [FAIL] {orbit}{fname} ({e})")

    print("\n=== Download task completed ===")


if __name__ == "__main__":
    main()
