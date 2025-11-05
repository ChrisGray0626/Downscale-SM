import os

# 文件后缀
TIFF_SUFFIX = ".tif"
NETCDF_SUFFIX = ".nc4"
HDF4_SUFFIX = ".hdf"
HDF5_SUFFIX = ".h5"
CSV_SUFFIX = ".csv"
PKL_SUFFIX = ".pkl"

# 空间范围：Left Bottom Right Top
RANGE = [-120, 35, -104, 49]

# 基础路径
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH: str = "/Volumes/Elements SE/Data/2016"
RESULT_PATH: str = os.path.join(ROOT_PATH, "Result")

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

TIFF_DIR_NAME = "tiff"
MERGED_DIR_NAME = "merged"
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

# 标准网格路径
STANDARD_GRID_1KM_PATH = os.path.join(RESULT_PATH, "Standard_Grid_1km.tif")
STANDARD_GRID_36KM_PATH = os.path.join(RESULT_PATH, "Standard_Grid_36km.tif")
# 标准网格分辨率
RESOLUTION_36KM = "36km"
RESOLUTION_1KM = "1km"

# 自变量列名
X_COLUMN = [
    NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME
]
# 因变量列名
Y_COLUMN = [SM_NAME]
SPATIAL_COLUMN = [PROJ_X_NAME, PROJ_Y_NAME]
