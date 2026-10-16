"""Reader for Juno/Waves PDS3 fixed-length binary tables (.lbl/.dat pairs).

The :class:`PDS3TableReader` parses the PDS3 label to obtain the table layout,
reads the fixed-length binary records of the companion ``.DAT`` file, splits the
records by receiver channel (LFR_LO / LFR_HI / HFR), and exports the content as
CSV or TXT files. E / B / NBS product types are supported.
"""

import re
import struct
import csv
import os
from pathlib import Path

import numpy as np


class PDS3TableReader:
    """Reader for Juno Waves PDS3 fixed-length binary tables.

    Supports E / B / NBS products and LFR_LO / LFR_HI / HFR records.
    """

    # =========================================================
    # 1. Initialization
    # =========================================================
    def __init__(self, lbl_path, dat_path=None):
        self.lbl_path = Path(lbl_path)
        self.dat_path = Path(dat_path) if dat_path else self.lbl_path.with_suffix(".DAT")

        self.lbl = self._read_lbl()
        self.layout = self._parse_layout()
        self.product_type = self._detect_product_type()

        self.records = self._read_dat()

    # =========================================================
    # 2. LBL parsing
    # =========================================================
    def _read_lbl(self):
        """Parse the PDS3 label file into a nested dictionary tree.

        OBJECT/END_OBJECT blocks become nested dictionaries under a synthetic
        ``__OBJECTS__`` key; scalar key/value pairs are converted to int or
        float when possible.
        """
        lbl = {}
        stack = [lbl]

        with open(self.lbl_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("/*"):
                    continue

                if line.startswith("OBJECT"):
                    _, name = line.split("=", 1)
                    obj = {"__OBJECT_NAME__": name.strip()}
                    stack[-1].setdefault("__OBJECTS__", []).append(obj)
                    stack.append(obj)
                    continue

                if line.startswith("END_OBJECT"):
                    stack.pop()
                    continue

                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()

                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]

                    if re.fullmatch(r"[+-]?\d+", v):
                        v = int(v)
                    elif re.fullmatch(r"[+-]?\d*\.\d+(E[+-]?\d+)?", v, re.I):
                        v = float(v)

                    stack[-1][k] = v

        return lbl

    def _find_objects(self, tree, name):
        """Recursively find all objects with the given ``__OBJECT_NAME__``."""
        found = []
        if isinstance(tree, dict):
            if tree.get("__OBJECT_NAME__") == name:
                found.append(tree)
            for v in tree.values():
                found.extend(self._find_objects(v, name))
        elif isinstance(tree, list):
            for x in tree:
                found.extend(self._find_objects(x, name))
        return found

    # =========================================================
    # 3. Layout extraction
    # =========================================================
    def _parse_layout(self):
        """Extract the record layout (record size, row count, column map)."""
        file_obj = self._find_objects(self.lbl, "FILE")[0]
        table = self._find_objects(file_obj, "DATA_TABLE")[0]
        columns = self._find_objects(table, "COLUMN")

        layout = {
            "record_bytes": file_obj["RECORD_BYTES"],
            "n_records": table["ROWS"],
            "columns": {},
        }

        for c in columns:
            name = c["NAME"]
            start = c["START_BYTE"] - 1  # PDS is 1-based
            size = c["BYTES"]

            layout["columns"][name] = {
                "start": start,
                "bytes": size,
                "type": c["DATA_TYPE"],
                "items": c.get("ITEMS"),
                "item_bytes": c.get("ITEM_BYTES"),
            }

        return layout

    # =========================================================
    # 4. Product detection (E / B / NBS)
    # =========================================================
    def _detect_product_type(self):
        """Infer the product type (E_FIELD / B_FIELD / NBS) from the file name."""
        name = self.dat_path.name.upper()
        if "_E_" in name:
            return "E_FIELD"
        if "_B_" in name:
            return "B_FIELD"
        if "_NBS_" in name:
            return "NBS"
        return "UNKNOWN"

    # =========================================================
    # 5. Scalar reader
    # =========================================================
    def _read_scalar(self, buf, col):
        """Decode a single scalar column value from a raw record buffer."""
        t = col["type"]
        b0 = col["start"]
        b1 = b0 + col["bytes"]

        if t == "CHARACTER":
            return buf[b0:b1].decode("ascii").strip()

        if t == "PC_REAL":
            return struct.unpack("<f" if col["bytes"] == 4 else "<d", buf[b0:b1])[0]

        if t == "LSB_UNSIGNED_INTEGER":
            return int.from_bytes(buf[b0:b1], "little")

        if t == "BOOLEAN":
            return buf[b0] != 0

        if t == "TIME":
            return buf[b0:b1].decode("ascii").strip()

        return None

    # =========================================================
    # 6. DAT record reader
    # =========================================================
    def _read_dat(self):
        """Read all data records from the .DAT file.

        The first record is a PDS3 header record and is skipped. The waveform
        column is decoded as little-endian single-precision floats and truncated
        to ``num_amplitudes`` samples.
        """
        L = self.layout
        C = L["columns"]
        recs = []

        with open(self.dat_path, "rb") as f:
            f.seek(L["record_bytes"])  # skip HEADER record

            for _ in range(L["n_records"]):
                buf = f.read(L["record_bytes"])
                if len(buf) < L["record_bytes"]:
                    break

                rec = {"product": self.product_type}

                for name, col in C.items():
                    if name == "WAVEFORM":
                        continue
                    rec[name.lower()] = self._read_scalar(buf, col)

                # Handle waveform.
                n = rec["num_amplitudes"]
                wf_col = C["WAVEFORM"]

                wf = np.frombuffer(
                    buf[wf_col["start"]: wf_col["start"] + wf_col["bytes"]],
                    dtype="<f4",
                ).ravel()[:n]

                rec["waveform"] = wf

                recs.append(rec)

        return recs

    # =========================================================
    # 7. Split LFR / HFR records
    # =========================================================
    def split_lfr_hfr(self):
        """Split records into LFR_LO, LFR_HI and HFR groups by channel name."""
        lfr_lo, lfr_hi, hfr = [], [], []

        for r in self.records:
            recv = r["channel"].upper()
            if recv.startswith("LFR_LO"):
                lfr_lo.append(r)
            elif recv.startswith("LFR_HI"):
                lfr_hi.append(r)
            elif recv.startswith("HFR"):
                hfr.append(r)

        return lfr_lo, lfr_hi, hfr

    # =========================================================
    # 8. Export
    # =========================================================
    def export(self, records, out_file, flatten_waveform=True):
        """Export records to a CSV or TXT file.

        Parameters
        ----------
        records : list of dict
            Records to export (e.g., one group from :meth:`split_lfr_hfr`).
        out_file : str
            Output path with a ``.csv`` or ``.txt`` extension.
        flatten_waveform : bool
            False -> waveform stored as a single space-separated string
                     under a ``waveform`` column.
            True  -> waveform stored as one column per sample
                     (``waveform_0``, ``waveform_1``, ...).
        """
        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if not records:
            print(f"[INFO] No records to export, skipping: {out_file}")
            return

        ext = os.path.splitext(out_file)[1].lower()
        if ext not in (".csv", ".txt"):
            raise ValueError("Only .csv or .txt supported")

        if ext == ".csv":
            self.export_csv(records, out_file, flatten_waveform)
        if ext == ".txt":
            self.export_txt(records, out_file, flatten_waveform)

    def export_csv(self, records, out_file, flatten_waveform):
        """Write records to CSV with optional flattened waveform columns."""
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            header = ["product", "channel", "trig_scet", "trig_sclk", "sampling_interval", "num_amplitudes"]
            if flatten_waveform:
                max_n = max(r["num_amplitudes"] for r in records)
                header += [f"wf_{i}" for i in range(max_n)]
            else:
                header.append("waveform")

            writer.writerow(header)

            for r in records:
                row = [
                    r["product"],
                    r["channel"],
                    r["trig_scet"],
                    r["trig_sclk"],
                    r["sampling_interval"],
                    r["num_amplitudes"],
                ]

                if flatten_waveform:
                    wf = list(r["waveform"]) + [""] * (max_n - r["num_amplitudes"])
                    row += wf
                else:
                    row.append(" ".join(f"{x:.6e}" for x in r["waveform"]))

                writer.writerow(row)

    def export_txt(self, records, out_file, flatten_waveform):
        """Write records to a plain-text file with metadata header lines."""
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(f"# channel = {r['channel']}\n")
                f.write(f"# trig_scet = {r['trig_scet']}\n")
                f.write(f"# trig_sclk = {r['trig_sclk']}\n")
                f.write(f"# sampling_interval = {r['sampling_interval']}\n")
                f.write(f"# num_amplitudes = {r['num_amplitudes']}\n")

                wf = " ".join(f"{x:.6e}" for x in r["waveform"])
                f.write(wf + "\n\n")

    def export_metadata_csv(self, records, out_file):
        """Export all record fields except the waveform to a CSV file."""
        if not records:
            print(f"[INFO] No records to export, skipping: {out_file}")
            return

        out_dir = os.path.dirname(out_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Get all keys except 'waveform'.
            sample = records[0]
            header = [k for k in sample.keys() if k != "waveform"]
            writer.writerow(header)

            for r in records:
                row = [r[k] for k in header]
                writer.writerow(row)
