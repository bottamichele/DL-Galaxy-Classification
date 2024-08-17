from galaxy_datasets import gz2

# ==================================================
# ================ GLOBAL VARIABLES ================
# ==================================================

PATH_GZ2_DATASET = "./gz2_dataset/"
GZ2_DATA_TABULAR_FILENAME = "gz2_hart16.csv"

# ========================================
# ========= DOWNLOAD GZ2 DATASET =========
# ========================================

if __name__ == "__main__":
    DOWNLOAD_NEEDED = False

    #Galaxy Zoo 2 is downloaded if needed.
    if DOWNLOAD_NEEDED:
        print("Galaxy Zoo 2 dataset is downloading...")
        gz2(root=PATH_GZ2_DATASET, train=False, download=True)