# preprocess

Preprocessing pipeline for Juno/Waves dust-impact analysis: downloading the PDS3
burst-mode waveform products and converting the `.dat`/`.lbl` pairs into CSV
tables ready for the CNN classifier (see the repository root `README.md`).

## Scripts

| Script | Purpose |
|---|---|
| `test_download09_05.py` | Crawls the PDS archive listing and downloads in-range Juno/Waves burst `.dat`/`.lbl` files. Configure the year/DOY window and local output root at the top of the file. |
| `scan_file_dir.py` | Scans a local product directory, groups `.LBL`/`.DAT` pairs by base name, and classifies products (E/B field, REC/BIN layout, NBS level). |
| `dat_lbl_file_reader.py` | `PDS3TableReader`: parses PDS3 labels, reads fixed-length binary `.DAT` records, splits LFR_LO/LFR_HI/HFR channels, and exports CSV/TXT/metadata tables. |
| `main_process.py` | Entry point: scans the local PDS3 directory, selects E-field products, and exports LFR_LO metadata CSVs. |
| `interpret_file.py` | Inspection helpers: reloads exported CSVs, plots waveforms/spectrograms, and normalizes waveform peaks. |

## Pipeline order

```
download  ->  scan_file_dir  ->  dat_lbl_file_reader  ->  main_process  ->  interpret_file
```

## Requirements

- Python 3.8+
- `requests`, `beautifulsoup4` (downloader)
- `numpy`, `scipy`, `matplotlib` (reader/plotting)

## Usage

```bash
# 1) Download burst waveform products (edit the config block first)
python test_download09_05.py

# 2) Convert .dat/.lbl products to CSV (edit parent_dir_scan first)
python main_process.py

# 3) Inspect exported products interactively
python -c "from interpret_file import read_exported_csv; print(read_exported_csv('Juno_WAVES_meta_csv/lfr_low/xxx_lfr_lo_metadata.csv'))"
```
