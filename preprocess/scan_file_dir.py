"""Utilities for scanning PDS3 product directories and classifying Juno/Waves products.

Provides :func:`scan_pds3_directory` to group ``.LBL``/``.DAT`` file pairs by base
name, :func:`classify_pds3_product` to infer product attributes from the file name,
and :func:`filter_products` to select products by those attributes.
"""

import os


def scan_pds3_directory(root_dir):
    """Recursively scan a directory and group PDS3 LBL/DAT files by base name.

    Each unique base name (file name without the extension) yields one entry
    carrying the directory path and the full paths of its ``.LBL`` and ``.DAT``
    files (``None`` if the counterpart is missing).

    Parameters
    ----------
    root_dir : str
        Root directory to scan recursively.

    Returns
    -------
    products : list of dict
        Each dict contains:
            base : filename prefix
            dir  : directory path
            LBL  : full path or None
            DAT  : full path or None

    Example
    -------
    >>> products = scan_pds3_directory("D:/JUNO/WAVES")
    >>> for p in products:
    ...     print(p["base"], p["LBL"], p["DAT"])
    """
    grouped = {}

    for path, _, files in os.walk(root_dir):
        for fname in files:
            if not (fname.endswith(".LBL") or fname.endswith(".DAT")):
                continue

            base = fname[:-4]
            full = os.path.join(path, fname)

            if base not in grouped:
                grouped[base] = {
                    "base": base,
                    "dir": path,
                    "LBL": None,
                    "DAT": None,
                }

            grouped[base][fname[-3:]] = full

    return list(grouped.values())


def classify_pds3_product(base_name):
    """Classify a PDS3 Waves product from its file name.

    Parameters
    ----------
    base_name : str
        File name without extension.

    Returns
    -------
    info : dict
        {
          "field":  "E" | "B" | None,
          "level":  "NBS" | "E" | "B" | None,
          "layout": "REC" | "BIN" | None
        }
    """
    info = {
        "field": None,   # E or B
        "level": None,   # NBS / E / B
        "layout": None,  # REC / BIN
    }

    name = base_name.upper()

    # -------- field type --------
    if "_E_" in name:
        info["field"] = "E"
    elif "_B_" in name:
        info["field"] = "B"

    # -------- layout --------
    if "_REC_" in name:
        info["layout"] = "REC"
    elif "_BIN_" in name:
        info["layout"] = "BIN"

    # -------- product level --------
    if "_NBS_" in name:
        info["level"] = "NBS"
    else:
        # For non-NBS products, use the E / B field as the engineering-level marker.
        info["level"] = info["field"]

    return info


def filter_products(products, *, field=None, level=None, layout=None):
    """Select products whose classification matches the given attributes.

    Parameters
    ----------
    products : list of dict
        Output of :func:`scan_pds3_directory`.
    field, level, layout : str, optional
        Attribute values to require; ``None`` means "do not filter on this attribute".

    Returns
    -------
    list of dict
        Subset of ``products`` matching all specified attributes.
    """
    out = []

    for p in products:
        cls = classify_pds3_product(p["base"])

        if field and cls["field"] != field:
            continue
        if level and cls["level"] != level:
            continue
        if layout and cls["layout"] != layout:
            continue

        out.append(p)

    return out
