import os

# 文件后缀
TIFF_SUFFIX = ".tif"
NETCDF_SUFFIX = ".nc4"
HDF4_SUFFIX = ".hdf"
HDF5_SUFFIX = ".h5"
CSV_SUFFIX = ".csv"
PKL_SUFFIX = ".pkl"
ZIP_SUFFIX = ".zip"

# 空间范围：Left Bottom Right Top
RANGE = [-120, 35, -104, 49]

# 基础路径
PROJ_PATH = os.getenv("PROJ_PATH") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR_PATH = os.path.join(PROJ_PATH, "CHECKPOINTS")
RESULT_DIR_PATH = os.path.join(PROJ_PATH, "RESULTS")
DATA_DIR_PATH = os.getenv("DATA_DIR_PATH", "/Volumes/Elements SE/Data/Downscale-DM")
RAW_DIR_PATH: str = os.path.join(DATA_DIR_PATH, "Raw/2016-2020")
PROCESSED_DIR_PATH = os.path.join(DATA_DIR_PATH, "Processed")
CONVERTED_DIR_NAME = "Converted"
MERGED_DIR_NAME = "Merged"
DECOMPRESSED_DIR_NAME = "Decompressed"
INFERENCE_DIR_PATH = os.path.join(DATA_DIR_PATH, "Inference")
CORRECTION_DIR_PATH = os.path.join(DATA_DIR_PATH, "Correction")

VALID_DATE_FILE_PATH = os.path.join(DATA_DIR_PATH, "ValidDate.txt")

# 数据名称
DATE_NAME = "Date"
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

# Reference Grid
REF_GRID_36KM_PATH = os.path.join(DATA_DIR_PATH, "Standard_Grid_36km.tif")
REF_GRID_1KM_PATH = os.path.join(DATA_DIR_PATH, "Standard_Grid_1km.tif")

RESOLUTION_36KM = "36km"
RESOLUTION_1KM = "1km"
