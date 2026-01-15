import zipfile

import numpy as np
import torch
from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin


def format_float(x, precision):
    if isinstance(x, float):
        rounded = round(x, precision)
        # 转成字符串后去除多余的0和小数点
        return float(
            str(rounded).rstrip('0').rstrip('.') if '.' in str(rounded) else str(rounded)
        )
    return x


def unzip_file(file_path, target_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)


def convert_projection(lons, lats, src_espg_code: int = 4326, dst_epsg_code: int = 6933):
    src_proj = f'EPSG:{src_espg_code}'
    dst_proj = f'EPSG:{dst_epsg_code}'
    transformer = Transformer.from_crs(src_proj, dst_proj, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    return xs, ys


def write_txt(dst_path, rows):
    with open(dst_path, 'w') as f:
        for row in rows:
            f.write(f"{row}\n")


def read_txt(src_path):
    with open(src_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def build_transform_from_lonlat(lons: np.ndarray, lats: np.ndarray) -> Affine:
    # Handle 2D regular grid
    if lons.ndim == 2:
        if np.allclose(lons, lons[0, :]):
            lons = lons[0, :]
        else:
            raise ValueError("lons is non-regular 2D grid, cannot directly generate transform")
    if lats.ndim == 2:
        if np.allclose(lats, lats[:, 0]):
            lats = lats[:, 0]
        else:
            raise ValueError("lats is non-regular 2D grid, cannot directly generate transform")

    # Check 1D
    if lons.ndim != 1 or lats.ndim != 1:
        raise ValueError("lons and lats must be 1D or 2D arrays to build transform")

    x_size = (lons.max() - lons.min()) / (len(lons) - 1)
    y_size = (lats.max() - lats.min()) / (len(lats) - 1)
    transform = from_origin(
        lons.min(),
        lats.max(),
        x_size,
        y_size,
    )

    return transform
