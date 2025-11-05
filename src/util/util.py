import os.path
import re
import zipfile
from datetime import datetime, timedelta

from pyproj import Transformer

from constant import *


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


def convert_year_day2date(year_day):
    # 解析年份和年积日
    year = int(year_day[:4])
    day_of_year = int(year_day[4:])

    # 构造当年的1月1日，然后加上年积日偏移
    date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)

    # 转换为 YYYYMMDD 格式
    return date.strftime('%Y%m%d')


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


def extract_date_from_modis_filename(filename: str):
    pattern = re.compile(r"^\w*?\.A(\d{7})\.h\d{2}v\d{2}\..*\.\w{3}$")
    match = pattern.match(filename)
    year_day = match.group(1)
    date = convert_year_day2date(year_day)

    return date


TGT_DATE_PATH = os.path.join(RESULT_PATH, "TGT_Date.txt")


def handle_tgt_date():
    dir_path = os.path.join(RESULT_PATH, NDVI_NAME, "1km")
    filenames = os.listdir(dir_path)
    # remove the suffix
    dates = [os.path.splitext(f)[0] for f in filenames if f.endswith(TIFF_SUFFIX)]
    dates = sorted(dates)

    write_txt(TGT_DATE_PATH, dates)


def is_tgt_date(date: str):
    tgt_dates = read_txt(TGT_DATE_PATH)
    return date in tgt_dates


def get_tgt_dates():
    tgt_dates = read_txt(TGT_DATE_PATH)
    return tgt_dates
