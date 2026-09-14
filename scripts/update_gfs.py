import datetime
import json
import math
import os
import sys
import numpy as np
import requests
import xarray as xr


def get_latest_gfs_info():
    """Tự động tìm ngày và cycle chạy GFS mới nhất vừa phát hành trên NOAA NOMADS."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for hours_back in [2, 5, 8, 11, 14, 17, 20]:
        check_time = now - datetime.timedelta(hours=hours_back)
        date_str = check_time.strftime('%Y%m%d')
        hour = check_time.hour
        cycle = f"{(hour // 6) * 6:02d}"

        # Link test sự tồn tại của bản tin
        url_test = (
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
            f"file=gfs.t{cycle}z.pgrb2.0p25.f000&lev_10_m_above_ground=on&var_UGRD=on&"
            f"subregion=&toplat=50&leftlon=90&rightlon=180&bottomlat=0&"
            f"dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
        )
        if requests.head(url_test).status_code == 200:
            return date_str, cycle

    return now.strftime('%Y%m%d'), '00'


def classify_system(max_wind_kt, min_mslp):
  """Phân loại hệ thống xoáy thuận nhiệt đới theo chuẩn quốc tế (Cấp gió WMO & Beaufort)."""
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


def detect_cyclones(ds_wind, ds_mslp, ds_850):
  """Thuật toán quét tự động phát hiện tâm xoáy Bão/ATNĐ dựa trên Độ xoáy (850hPa),

  Cực tiểu MSLP và Vận tốc gió surface.
  """
  detected_systems = []

  try:
    lats = ds_wind['latitude'].values
    lons = ds_wind['longitude'].values

    mslp = ds_mslp['prmsl'].values / 100.0  # Pa -> hPa
    u10 = ds_wind['u10'].values
    v10 = ds_wind['v10'].values
    wind_speed_10m = np.sqrt(u10**2 + v10**2)  # m/s

    # Lấy độ xoáy tầng 850hPa (Abs Vorticity)
    vort850 = ds_850['absv'].values if 'absv' in ds_850 else None

    # Tìm các cực tiểu áp suất cục bộ (Local minima MSLP)
    # Quét qua lưới dữ liệu (bỏ biên)
    for i in range(5, len(lats) - 5, 4):
      for j in range(5, len(lons) - 5, 4):
        center_mslp = mslp[i, j]

        # Điều kiện 1: Áp suất trung tâm phải thấp (<= 1008 hPa)
        if center_mslp <= 1008.0:
          # Kiểm tra xem đây có phải cực tiểu trong bán kính khoảng 300km (12 lưới)
          sub_mslp = mslp[max(0, i - 6) : i + 7, max(0, j - 6) : j + 7]
          if center_mslp == np.min(sub_mslp):
            # Lấy cực đại gió 10m trong bán kính xung quanh tâm
            sub_wind = wind_speed_10m[
                max(0, i - 8) : i + 9, max(0, j - 8) : j + 9
            ]
            max_wind_ms = np.max(sub_wind) if sub_wind.size > 0 else 0
            max_wind_kts = max_wind_ms * 1.94384  # m/s -> Knots

            # Điều kiện 2: Phải có xoáy (Gió max xung quanh >= 10 m/s ~ 20 knots)
            if max_wind_ms >= 10.0 or center_mslp <= 1002.0:
              lat_val = float(lats[i])
              lon_val = float(lons[j])

              system_info = classify_system(max_wind_kts, center_mslp)

              detected_systems.append({
                  'lat': round(lat_val, 2),
                  'lon': round(lon_val, 2),
                  'min_mslp_hpa': round(float(center_mslp), 1),
                  'max_wind_kts': round(float(max_wind_kts), 1),
                  'max_wind_ms': round(float(max_wind_ms), 1),
                  'type': system_info['type'],
                  'label': system_info['label'],
                  'category': system_info['category'],
              })
  except Exception as e:
    print(f'⚠️ Lỗi trong quá trình quét bão: {e}')

  return detected_systems


# --- CHƯƠNG TRÌNH CHÍNH ---
date_str, cycle = get_latest_gfs_info()
print(f'-> Đã xác định bản tin GFS mới nhất: Ngày {date_str} - Cycle {cycle}Z')

# Danh sách mốc dự báo (0h đến 24h)
forecast_hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
time_series_data = []
base_time = datetime.datetime.strptime(f'{date_str}{cycle}', '%Y%m%d%H')

for f_hr in forecast_hours:
  f_str = f'{f_hr:03d}'

  # Link NOAA bổ sung thêm biến tầng 850hPa (Độ xoáy ABSV) để nhận diện bão
  url = (
      f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
      f'file=gfs.t{cycle}z.pgrb2.0p25.f{f_str}&'
      f'lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&'
      f'lev_mean_sea_level=on&var_PRMSL=on&'
      f'lev_850_mb=on&var_ABSV=on&'
      f'subregion=&toplat=47&leftlon=87&rightlon=180&bottomlat=0&'
      f'dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos'
  )
  grib_file = f'gfs_f{f_str}.grib2'

  res = requests.get(url, stream=True)
  with open(grib_file, 'wb') as f:
    for chunk in res.iter_content(chunk_size=8192):
      f.write(chunk)

  try:
    # 1. Đọc tầng Surface (Gió 10m)
    ds_wind = xr.open_dataset(
        grib_file,
        engine='cfgrib',
        filter_by_keys={'typeOfLevel': 'heightAboveGround'},
    )
    # 2. Đọc tầng MSLP (Áp suất mực nước biển)
    ds_mslp = xr.open_dataset(
        grib_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'meanSea'}
    )
    # 3. Đọc tầng 850hPa (Độ xoáy phục vụ phân tích bão)
    ds_850 = xr.open_dataset(
        grib_file,
        engine='cfgrib',
        filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'level': 850},
    )

    u, v = ds_wind['u10'].values, ds_wind['v10'].values
    mslp = ds_mslp['prmsl'].values / 100.0  # Pa -> hPa
    lats, lons = ds_wind['latitude'].values, ds_wind['longitude'].values

    # Chạy thuật toán tự động quét Bão / Áp thấp nhiệt đới
    cyclones = detect_cyclones(ds_wind, ds_mslp, ds_850)

    frame_time = base_time + datetime.timedelta(hours=f_hr)

    time_series_data.append({
        'forecastTime': frame_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'forecastHour': f_hr,
        'cyclonesDetected': cyclones,  # Mảng danh sách các cơn Bão/ATNĐ phát hiện được
        'data': [
            # Layer 0: Gió U (10m)
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
                    for val in u.flatten()
                ],
            },
            # Layer 1: Gió V (10m)
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
                    for val in v.flatten()
                ],
            },
            # Layer 2: Áp suất MSLP (hPa)
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

    ds_wind.close()
    ds_mslp.close()
    ds_850.close()
    os.remove(grib_file)
    print(
        f'-> Xử lý xong mốc +{f_hr}h (Phát hiện {len(cyclones)} hệ thống áp'
        ' thấp/bão)'
    )
  except Exception as e:
    print(f'❌ Lỗi mốc f{f_str}: {e}')

# Ghi kết quả ra thư mục public/data
output_dir = 'public/data'
os.makedirs(output_dir, exist_ok=True)
output_filepath = os.path.join(output_dir, 'gfs_nwp_timeseries.json')

with open(output_filepath, 'w') as f:
  json.dump(time_series_data, f)

print(f'✅ ĐÃ XUẤT THÀNH CÔNG FILE DATA: {output_filepath}')
