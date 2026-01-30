import os

# 文件后缀
TIFF_SUFFIX = ".tif"
NETCDF_SUFFIX = ".nc"
NETCDF4_SUFFIX = ".nc4"
HDF4_SUFFIX = ".hdf"
HDF5_SUFFIX = ".h5"
CSV_SUFFIX = ".csv"
PKL_SUFFIX = ".pkl"
ZIP_SUFFIX = ".zip"

# 空间范围：Left Bottom Right Top
RANGE = [-120, 35, -104, 49]

DDPM_NAME = "DDPM"
DDPM_MODEL_PATH = os.path.join("CHECKPOINTS", DDPM_NAME)

# 基础路径
PROJ_PATH = os.getenv("PROJ_PATH") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR_PATH = os.path.join(PROJ_PATH, "CHECKPOINTS")
RESULT_DIR_PATH = os.path.join(PROJ_PATH, "RESULTS")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH", "/Volumes/Elements SE/Data/SMDownscaling")
RAW_NAME = "Raw"
RAW_DIR_PATH = os.path.join(DATA_DIR_PATH, "Raw/2016-2020")
PROCESSED_NAME = "Processed"
PROCESSED_DIR_PATH = os.path.join(DATA_DIR_PATH, PROCESSED_NAME)
CONVERTED_DIR_NAME = "Converted"
INTERPOLATED_DIR_NAME = "Interpolated"
MERGED_DIR_NAME = "Merged"
DECOMPRESSED_DIR_NAME = "Decompressed"
INFERENCE_DIR_PATH = os.path.join(DATA_DIR_PATH, "Inference")
CORRECTION_DIR_PATH = os.path.join(DATA_DIR_PATH, "Correction")

VALID_DATE_FILE_PATH = os.path.join(DATA_DIR_PATH, "VALID_DATES.txt")

# 数据名称
X_NAME = "X"
Y_NAME = "Y"
DATE_NAME = "Date"
POS_NAME = "Pos"
LONGITUDE_NAME = "Lon"
LATITUDE_NAME = "Lat"
PROJ_X_NAME = "ProjX"
PROJ_Y_NAME = "ProjY"
ROW_NAME = "Row"
COL_NAME = "Col"
DATA_NAME = "Data"
NDVI_NAME = "NDVI"
LST_NAME = "LST"
SM_NAME = "SM"
ALBEDO_NAME = "Albedo"
PRECIPITATION_NAME = "Precipitation"
DEM_NAME = "DEM"
IN_SITU_NAME = "InSitu"
ESA_CCI_NAME = "ESA-CCI"
SMAP_HB_NAME = "SMAP-HB"
RESOLUTION_36KM = "36km"
RESOLUTION_25KM = "25km"
RESOLUTION_1KM = "1km"

# Reference Grid
REF_GRID_36KM_PATH = os.path.join(DATA_DIR_PATH, f"REF_GRID_{RESOLUTION_36KM}{TIFF_SUFFIX}")
REF_GRID_25KM_PATH = os.path.join(DATA_DIR_PATH, f"REF_GRID_{RESOLUTION_25KM}{TIFF_SUFFIX}")
REF_GRID_1KM_PATH = os.path.join(DATA_DIR_PATH, f"REF_GRID_{RESOLUTION_1KM}{TIFF_SUFFIX}")

RF_NAME = "RF"
RF_DIR_PATH = os.path.join(DATA_DIR_PATH, RF_NAME)
RF_MODEL_PATH = os.path.join(CHECKPOINT_DIR_PATH, RF_NAME, "model.pkl")

GWR_NAME = "GWR"
GWR_DIR_PATH = os.path.join(DATA_DIR_PATH, GWR_NAME)
GWR_MODEL_PATH = os.path.join(CHECKPOINT_DIR_PATH, GWR_NAME, "model.pkl")

RESNET_NAME = "ResNet"
RESNET_DIR_PATH = os.path.join(DATA_DIR_PATH, RESNET_NAME)
RESNET_MODEL_PATH = os.path.join(CHECKPOINT_DIR_PATH, RESNET_NAME, "model.pt")
