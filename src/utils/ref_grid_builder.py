import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from constants import RANGE, RESULT_DIR_PATH

# TODO
# 网格分辨率（单位：米）
resolution = 1000

output_file = "Standard_Grid_36km.tif"
output_path = RESULT_DIR_PATH + output_file

# 设置经纬度空间范围
lon_min, lat_min, lon_max, lat_max = RANGE
# 坐标转换：WGS84 → EASE-Grid 2.0 (EPSG:6933)
wgs84_crs = CRS.from_epsg(4326)
ease_crs = CRS.from_epsg(6933)
transformer = Transformer.from_crs(wgs84_crs, ease_crs, always_xy=True)
# 边界四角点投影
x_min, y_max = transformer.transform(lon_min, lat_max)
x_max, y_min = transformer.transform(lon_max, lat_min)
# 对齐网格边界到格点（向外扩展）
x_min_aligned = np.floor(x_min / resolution) * resolution
x_max_aligned = np.ceil(x_max / resolution) * resolution
y_min_aligned = np.floor(y_min / resolution) * resolution
y_max_aligned = np.ceil(y_max / resolution) * resolution
# 构建网格坐标
x_coords = np.arange(x_min_aligned, x_max_aligned + resolution, resolution)
y_coords = np.arange(y_max_aligned, y_min_aligned - resolution, -resolution)
nrows = len(y_coords)
ncols = len(x_coords)
# 构建填充数据
fill_values = np.full((nrows, ncols), np.nan, dtype=np.float32)
# 构建 affine transform（左上角原点）
transform = from_origin(x_min_aligned, y_max_aligned, resolution, resolution)
# 写入 GeoTIFF
with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=nrows,
        width=ncols,
        count=1,
        dtype=fill_values.dtype,
        crs=ease_crs.to_string(),
        transform=transform,
        nodata=np.nan
) as dst:
    dst.write(fill_values, 1)
