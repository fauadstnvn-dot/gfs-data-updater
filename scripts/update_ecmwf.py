import datetime
import json
import math
import os
import numpy as np
import xarray as xr
from ecmwf.opendata import Client


def classify_system(max_wind_kt, min_mslp):
  """Phân loại hệ thống xoáy thuận nhiệt đới."""
  if max_wind_kt >= 64:
    return {'type': 'TYPHOON', 'label': 'Bão mạnh / Typhoon', 'category': 3}
  elif max_wind_kt >= 34:
    return {'type': 'TROPICAL_STORM', 'label': 'Bão nhiệt đới', 'category': 2}
  elif max_wind_kt >= 22 or min_mslp <= 1004:
    return {
        'type': 'TROPICAL_DEPRESSION',
        'label': 'Áp thấp nhiệt đới',
        'category': 1,
    }
  else:
    return {'type': 'LOW_PRESSURE', 'label': 'Vùng áp thấp', 'category': 0}


def detect_cyclones(mslp_grid, u10_grid, v10_grid, lats, lons):
  """Quét phát hiện bão/ATNĐ trên lưới ECMWF."""
  detected_systems = []
  try:
    wind_speed = np.sqrt(u10_grid**2 + v10_grid**2)

    for i in range(5, len(lats) - 5, 4):
      for j in range(5, len(lons) - 5, 4):
        center_mslp = mslp_grid[i, j]

        if center_mslp <= 1008.0:
          sub_mslp = mslp_grid[max(0, i - 6) : i + 7, max(0, j - 6) : j + 7]
          if center_mslp == np.min(sub_mslp):
            sub_wind = wind_speed[max(0, i - 8) : i + 9, max(0, j - 8) : j + 9]
            max_wind_ms = np.max(sub_wind) if sub_wind.size > 0 else 0
            max_wind_kts = max_wind_ms * 1.94384

            if max_wind_ms >= 10.0 or center_mslp <= 1002.0:
              sys_info = classify_system(max_wind_kts, center_mslp)
              detected_systems.append({
                  'lat': round(float(lats[i]), 2),
                  'lon': round(float(lons[j]), 2),
                  'min_mslp_hpa': round(float(center_mslp), 1),
                  'max_wind_kts': round(float(max_wind_kts), 1),
                  'max_wind_ms': round(float(max_wind_ms), 1),
                  'type': sys_info['type'],
                  'label': sys_info['label'],
                  'category': sys_info['category'],
              })
  except Exception as e:
    print(f'⚠️ Lỗi quét bão ECMWF: {e}')

  return detected_systems


# --- CHƯƠNG TRÌNH CHÍNH ECMWF ---
print('-> Bắt đầu kết nối tải dữ liệu ECMWF Open Data...')
client = Client(source='ecmwf', model='ifs', resol='0p25')

# Các mốc dự báo ECMWF (0h đến 24h, bước nhảy 3 tiếng)
forecast_steps = [0, 3, 6, 9, 12, 15, 18, 21, 24]
time_series_data = []

# Phạm vi khu vực Tây Bắc Thái Bình Dương: [N, W, S, E]
area_crop = [47, 83, 0, 180]

for step in forecast_steps:
  grib_filename = f'ecmwf_step_{step}.grib2'

  try:
    # Tải biến Gió 10m (10u, 10v) và Áp suất MSLP (msl) từ ECMWF
    client.retrieve(
        datetime=0,  # Tự lấy bản tin mới nhất khả dụng
        stream='oper',
        type='fc',
        step=step,
        param=['10u', '10v', 'msl'],
        target=grib_filename,
    )

    ds = xr.open_dataset(grib_filename, engine='cfgrib')

    # Báo crop tọa độ khu vực
    ds_sub = ds.sel(
        latitude=slice(area_crop[0], area_crop[2]),
        longitude=slice(area_crop[1], area_crop[3]),
    )

    lats = ds_sub['latitude'].values
    lons = ds_sub['longitude'].values
    u10 = ds_sub['u10'].values
    v10 = ds_sub['v10'].values
    mslp = ds_sub['msl'].values / 100.0  # Pa -> hPa

    # Lấy mốc thời gian dự báo
    valid_time = str(ds_sub['valid_time'].values)[:19] + 'Z'

    # Quét bão
    cyclones = detect_cyclones(mslp, u10, v10, lats, lons)

    time_series_data.append({
        'forecastTime': valid_time,
        'forecastHour': step,
        'cyclonesDetected': cyclones,
        'data': [
            # Layer 0: U10
            {
                'header': {
                    'parameterCategory': 2,
                    'parameterNumber': 2,
                    'nx': int(len(lons)),
                    'ny': int(len(lats)),
                    'lo1': float(lons[0]),
                    'la1': float(lats[0]),
                    'lo2': float(lons[-1]),
                    'la2': float(lats[-1]),
                    'dx': 0.25,
                    'dy': 0.25,
                },
                'data': [
                    round(float(val), 1) if not math.isnan(val) else 0.0
                    for val in u10.flatten()
                ],
            },
            # Layer 1: V10
            {
                'header': {
                    'parameterCategory': 2,
                    'parameterNumber': 3,
                    'nx': int(len(lons)),
                    'ny': int(len(lats)),
                    'lo1': float(lons[0]),
                    'la1': float(lats[0]),
                    'lo2': float(lons[-1]),
                    'la2': float(lats[-1]),
                    'dx': 0.25,
                    'dy': 0.25,
                },
                'data': [
                    round(float(val), 1) if not math.isnan(val) else 0.0
                    for val in v10.flatten()
                ],
            },
            # Layer 2: MSLP
            {
                'header': {
                    'parameterCategory': 3,
                    'parameterNumber': 1,
                    'nx': int(len(lons)),
                    'ny': int(len(lats)),
                    'lo1': float(lons[0]),
                    'la1': float(lats[0]),
                    'lo2': float(lons[-1]),
                    'la2': float(lats[-1]),
                    'dx': 0.25,
                    'dy': 0.25,
                },
                'data': [
                    round(float(val), 1) if not math.isnan(val) else 1013.25
                    for val in mslp.flatten()
                ],
            },
        ],
    })

    ds.close()
    os.remove(grib_filename)
    print(f'-> ECMWF: Đã xử lý xong mốc +{step}h')

  except Exception as e:
    print(f'❌ Lỗi mốc ECMWF step {step}: {e}')

# Xuất ra file json riêng cho ECMWF
output_dir = 'public/data'
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, 'ecmwf_nwp_timeseries.json')

with open(output_filepath, 'w') as f:
  json.dump(time_series_data, f)

print(f'✅ ĐÃ XUẤT THÀNH CÔNG ECMWF JSON: {output_filepath}')
