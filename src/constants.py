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
PROJ_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR_PATH = "/Volumes/Elements SE/Data/Downscale-DM"
RAW_DIR_PATH: str = os.path.join(DATA_DIR_PATH, "Raw/2016-2020")
PROCESSED_DIR_PATH = os.path.join(DATA_DIR_PATH, "Processed")
RESULT_PATH: str = os.path.join(PROJ_PATH, "Result")

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
LABELED_DATA_NAME = "LabeledData"
PRED_DATA_NAME = "PredData"

CONVERTED_DIR_NAME = "Converted"
MERGED_DIR_NAME = "Merged"
DECOMPRESSED_DIR_NAME = "Decompressed"

# 输入数据路径
LABELED_DATA_DIR_PATH = os.path.join(RESULT_PATH, LABELED_DATA_NAME)
# 预测数据路径
PRED_DATA_DIR_PATH = os.path.join(RESULT_PATH, PRED_DATA_NAME)
# 预测结果路径
PRED_RESULT_DIR_PATH = os.path.join(RESULT_PATH, "PredResult")
# 模型路径
GNNWR_DIR = os.path.join(RESULT_PATH, "GNNWR")
MODEL_NAME = "GNNWR_SM"
MODEL_DIR_PATH = os.path.join(GNNWR_DIR, "Model")
# 数据集路径
DATASET_DIR_PATH = os.path.join(GNNWR_DIR, "Dataset")
TRAIN_DATASET_DIR_PATH = os.path.join(DATASET_DIR_PATH, "train_dataset")
VAL_DATASET_DIR_PATH = os.path.join(DATASET_DIR_PATH, "val_dataset")
TEST_DATASET_DIR_PATH = os.path.join(DATASET_DIR_PATH, "test_dataset")
# 模型文件路径
MODEL_FILE_PATH = os.path.join(MODEL_DIR_PATH, f"{MODEL_NAME}{PKL_SUFFIX}")

# Reference Grid
REF_GRID_1KM_PATH = os.path.join(RESULT_PATH, "Standard_Grid_1km.tif")
REF_GRID_36KM_PATH = os.path.join(RESULT_PATH, "Standard_Grid_36km.tif")
# Reference Grid Resolution
RESOLUTION_36KM = "36km"
RESOLUTION_1KM = "1km"
