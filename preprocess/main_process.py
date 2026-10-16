"""Entry point of the Juno/Waves DAT-to-CSV preprocessing pipeline.

Scans a local PDS3 product directory for Juno Waves E-field products, parses each
product's .lbl/.dat pair with :class:`PDS3TableReader`, and exports per-receiver
metadata tables as CSV files.
"""

from scan_file_dir import scan_pds3_directory, filter_products
from dat_lbl_file_reader import PDS3TableReader


def dat2csv_E_field(base, out_dir, lbl_dir_filename, dat_dir_filename=None):
    """Parse one E-field PDS3 product and export its LFR_LO metadata to CSV.

    Parameters
    ----------
    base : str
        Product base name (file name without extension).
    out_dir : str
        Output directory; receiver-specific subdirectories are created inside it.
    lbl_dir_filename : str
        Full path to the .lbl label file.
    dat_dir_filename : str, optional
        Full path to the .dat data file. Defaults to the .lbl path with a .DAT
        extension (see :class:`PDS3TableReader`).
    """
    reader = PDS3TableReader(lbl_dir_filename, dat_dir_filename)

    lfr_lo, lfr_hi, hfr = reader.split_lfr_hfr()

    reader.export_metadata_csv(lfr_lo, f"{out_dir}/lfr_low/{base}_lfr_lo_metadata.csv")


def main():
    # Output directory (relative to the working directory).
    export_dir = "Juno_WAVES_meta_csv"

    # Root directory containing the PDS3 Jupiter-orbit products to be processed.
    parent_dir_scan = "I:/Zhangyq12212839/Juno_Dust/dust_data_dat/Jupiter_ORBIT"

    file_dir_names = scan_pds3_directory(parent_dir_scan)

    # Keep only electric-field (E) products.
    juno_waves_e_data_file_dir_names = filter_products(file_dir_names, field="E")

    for product in juno_waves_e_data_file_dir_names:
        base = product["base"]
        lbl_file = product["LBL"]
        print(base)
        dat2csv_E_field(base, export_dir, lbl_file)


if __name__ == "__main__":
    main()
