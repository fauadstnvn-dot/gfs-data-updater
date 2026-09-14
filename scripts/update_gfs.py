import datetime
import json
import math
import os
import requests
import xarray as xr


def get_latest_gfs_info():
  now = datetime.datetime.now(datetime.timezone.utc)
  for hours_back in [2, 5, 8, 11, 14, 17, 20]:
    check_time = now - datetime.timedelta(hours=hours_back)
    date_str = check_time.strftime('%Y%m%d')
    hour = check_time.hour
    cycle = f'{(hour // 6) * 6:02d}'
    url_test = (
        f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
        f'file=gfs.t{cycle}z.pgrb2.0p25.f000&lev_10_m_above_ground=on&var_UGRD=on&'
        f'subregion=&toplat=50&leftlon=90&rightlon=180&bottomlat=0&'
        f'dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos'
    )
    if requests.head(url_test).status_code == 200:
      return date_str, cycle
  return now.strftime('%Y%m%d'), '00'


date_str, cycle = get_latest_gfs_info()
print(f'-> Đã tìm thấy bản tin GFS mới nhất: Ngày {date_str} - Cycle {cycle}Z')

# 9 mốc thời gian: từ hiện tại (+0h) đến +24h
forecast_hours = [0, 3, 6, 9, 12, 15, 18, 21, 24]
time_series_data = []
base_time = datetime.datetime.strptime(f'{date_str}{cycle}', '%Y%m%d%H')

for f_hr in forecast_hours:
  f_str = f'{f_hr:03d}'
  url = (
      f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?'
      f'file=gfs.t{cycle}z.pgrb2.0p25.f{f_str}&'
      f'lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&'
      f'lev_mean_sea_level=on&var_PRMSL=on&'
      f'subregion=&toplat=50&leftlon=90&rightlon=180&bottomlat=0&'
      f'dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos'
  )
  grib_file = f'gfs_f{f_str}.grib2'

  res = requests.get(url, stream=True)
  with open(grib_file, 'wb') as f:
    for chunk in res.iter_content(chunk_size=8192):
      f.write(chunk)

  try:
    ds_wind = xr.open_dataset(
        grib_file,
        engine='cfgrib',
        filter_by_keys={'typeOfLevel': 'heightAboveGround'},
    )
    ds_mslp = xr.open_dataset(
        grib_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'meanSea'}
    )

    u, v = ds_wind['u10'].values, ds_wind['v10'].values
    mslp = ds_mslp['prmsl'].values / 100.0
    lats, lons = ds_wind['latitude'].values, ds_wind['longitude'].values

    frame_time = base_time + datetime.timedelta(hours=f_hr)

    time_series_data.append({
        'forecastTime': frame_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'forecastHour': f_hr,
        'data': [
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
    os.remove(grib_file)
    print(f'-> Xử lý xong mốc +{f_hr}h')
  except Exception as e:
    print(f'Lỗi mốc f{f_str}: {e}')

output_dir = 'public/data'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'gfs_nwp_timeseries.json'), 'w') as f:
  json.dump(time_series_data, f)

print('✅ Đã xuất file JSON thành công!')
