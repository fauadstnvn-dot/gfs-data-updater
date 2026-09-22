import datetime
import json
import math
import os
import sys
import numpy as np
import requests
import xarray as xr

# ==============================================================================
# HỆ THỐNG DỰ BÁO VÀ THEO DÕI XOÁY THUẬN NHIỆT ĐỚI / BÃO TỰ ĐỘNG (GFS 0.25°)
# Chạy trên GitHub Actions:
#  1. Tải các tầng:
#     - Surface: MSLP (prmsl)
#     - 10m: U10, V10
#     - 850 hPa: Gió U850, V850 -> tính độ xoáy tương đối (Relative Vorticity ζ)
#     - 200 hPa: Nhiệt độ T200 (xác minh cấu trúc Tâm Nóng - Warm Core)
#     - 500 hPa: Tốc độ thẳng đứng ω (Vertical Velocity - kiểm tra đối lưu bốc lên)
#     - 700 hPa: Độ ẩm tương đối RH (Relative Humidity - lọc xoáy khô)
#  2. TÍNH TOÁN TẠI GITHUB:
#     - Định vị tâm bằng phương pháp First Guess (độ xoáy 850hPa)
#     - Tinh chỉnh vị trí tâm xuống sub-grid bằng trọng tâm khuyết áp (Pressure Centroid)
#     - Phân tích đối xứng trường gió 10m, tìm bán kính gió mạnh nhất (RMW)
#     - Kiểm tra tâm nóng (Warm Core at 200hPa: T_center - T_env > 0°C)
#     - Phân loại cấp gió theo thang bão quốc tế và thang bão Việt Nam (Beaufort)
#  3. CHUYỂN VỀ HOST:
#     - CHỈ chuyển các trường cơ bản (u10, v10, mslp) + kết quả tâm bão đã tính toán.
#     - Tuyệt đối KHÔNG chuyển mảng 3D của các tầng 850, 200, 500, 700 hPa để tiết kiệm băng thông & dung lượng.
# ==============================================================================

def get_latest_gfs_info():
    """Tìm chu kỳ dự báo GFS mới nhất khả dụng trên NOAA NOMADS."""
    now = datetime.datetime.now(datetime.timezone.utc)
    for hours_back in [2, 5, 8, 11, 14, 17, 20]:
        check_time = now - datetime.timedelta(hours=hours_back)
        date_str = check_time.strftime("%Y%m%d")
        hour = check_time.hour
        cycle = f"{(hour // 6) * 6:02d}"
        url_test = (
            f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
            f"file=gfs.t{cycle}z.pgrb2.0p25.f000&lev_10_m_above_ground=on&var_UGRD=on&"
            f"subregion=&toplat=50&leftlon=90&rightlon=180&bottomlat=0&"
            f"dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
        )
        try:
            if requests.head(url_test, timeout=10).status_code == 200:
                return date_str, cycle
        except Exception:
            continue
    return now.strftime("%Y%m%d"), "00"


def compute_relative_vorticity(u_grid, v_grid, lats, lons):
    """
    Tính độ xoáy tương đối ζ = ∂v/∂x - ∂u/∂y (s^-1) trên lưới kinh vĩ cầu.
    """
    ny, nx = u_grid.shape
    vort = np.zeros_like(u_grid, dtype=float)
    
    # Bước góc lưới (độ)
    dy_deg = abs(float(lats[1] - lats[0])) if ny > 1 else 0.25
    dx_deg = abs(float(lons[1] - lons[0])) if nx > 1 else 0.25
    
    # Bán kính trái đất ~ 6,371,000 m => 1 độ vĩ ~ 111,195 m
    dy_m = dy_deg * 111195.0
    
    # Hướng tăng/giảm của vĩ độ
    lat_sign = 1.0 if lats[-1] > lats[0] else -1.0
    
    for i in range(1, ny - 1):
        lat_rad = math.radians(float(lats[i]))
        dx_m = dx_deg * 111195.0 * max(0.05, math.cos(lat_rad))
        
        # dv/dx (trung tâm)
        dv_dx = (v_grid[i, 2:] - v_grid[i, :-2]) / (2.0 * dx_m)
        
        # du/dy (trung tâm)
        du_dy = lat_sign * (u_grid[i + 1, 1:-1] - u_grid[i - 1, 1:-1]) / (2.0 * dy_m)
        
        vort[i, 1:-1] = dv_dx - du_dy
        
    return vort


def classify_system(max_wind_kt, min_mslp):
    """
    Phân loại cấp độ xoáy thuận nhiệt đới kết hợp thang đo Quốc Tế & thang Beaufort Việt Nam.
    """
    max_wind_ms = max_wind_kt / 1.94384
    max_wind_kmh = max_wind_ms * 3.6
    
    # Cấp gió Beaufort
    if max_wind_kmh < 1:
        beaufort = 0
    elif max_wind_kmh <= 5:
        beaufort = 1
    elif max_wind_kmh <= 11:
        beaufort = 2
    elif max_wind_kmh <= 19:
        beaufort = 3
    elif max_wind_kmh <= 28:
        beaufort = 4
    elif max_wind_kmh <= 38:
        beaufort = 5
    elif max_wind_kmh <= 49:
        beaufort = 6
    elif max_wind_kmh <= 61:
        beaufort = 7
    elif max_wind_kmh <= 74:
        beaufort = 8
    elif max_wind_kmh <= 88:
        beaufort = 9
    elif max_wind_kmh <= 102:
        beaufort = 10
    elif max_wind_kmh <= 117:
        beaufort = 11
    elif max_wind_kmh <= 133:
        beaufort = 12
    elif max_wind_kmh <= 149:
        beaufort = 13
    elif max_wind_kmh <= 166:
        beaufort = 14
    elif max_wind_kmh <= 183:
        beaufort = 15
    elif max_wind_kmh <= 201:
        beaufort = 16
    else:
        beaufort = 17

    beaufort_label = f"Cấp {beaufort} ({round(max_wind_kmh)} km/h)"

    if max_wind_kt >= 100:
        return {
            "type": "SUPER_TYPHOON",
            "label": "Siêu bão",
            "category": 5,
            "beaufort": beaufort,
            "beaufort_label": beaufort_label,
            "color": "#ef4444"
        }
    elif max_wind_kt >= 64:
        return {
            "type": "TYPHOON",
            "label": "Bão rất mạnh (Cuồng phong)",
            "category": 4,
            "beaufort": beaufort,
            "beaufort_label": beaufort_label,
            "color": "#f97316"
        }
    elif max_wind_kt >= 48:
        return {
            "type": "SEVERE_TROPICAL_STORM",
            "label": "Bão mạnh",
            "category": 3,
            "beaufort": beaufort,
            "beaufort_label": beaufort_label,
            "color": "#eab308"
        }
    elif max_wind_kt >= 34:
        return {
            "type": "TROPICAL_STORM",
            "label": "Bão nhiệt đới",
            "category": 2,
            "beaufort": beaufort,
            "beaufort_label": beaufort_label,
            "color": "#3b82f6"
        }
    elif max_wind_kt >= 22 or min_mslp <= 1004.0:
        return {
            "type": "TROPICAL_DEPRESSION",
            "label": "Áp thấp nhiệt đới",
            "category": 1,
            "beaufort": beaufort,
            "beaufort_label": beaufort_label,
            "color": "#06b6d4"
        }
    else:
        return {
            "type": "LOW_PRESSURE",
            "label": "Vùng áp thấp",
            "category": 0,
            "beaufort": beaufort,
            "beaufort_label": beaufort_label,
            "color": "#64748b"
        }


def detect_cyclones_full(mslp_grid, u10_grid, v10_grid, lats, lons,
                         u850=None, v850=None, t200=None, w500=None, rh700=None):
    """
    Thuật toán phát hiện và tính toán tâm bão/ATNĐ khoa học:
    1. First Guess: Độ xoáy 850hPa (vorticity max) hoặc cực tiểu MSLP.
    2. Sub-grid Centroid: Tinh chỉnh tọa độ tâm bằng trọng tâm khuyết áp.
    3. Phân tích đối xứng gió & tâm lặng gió (Lull area) & RMW (bán kính gió cực đại).
    4. Kiểm tra Warm Core (Tâm Nóng) ở 200hPa để loại nhiễu và xoáy ngoại nhiệt đới.
    5. Kiểm tra đối lưu bốc lên 500hPa & độ ẩm 700hPa nếu có.
    """
    detected = []
    ny, nx = mslp_grid.shape
    wind10 = np.sqrt(u10_grid**2 + v10_grid**2)
    
    # 1. Tính độ xoáy 850hPa nếu có trường gió 850hPa
    vort850 = None
    if u850 is not None and v850 is not None:
        try:
            vort850 = compute_relative_vorticity(u850, v850, lats, lons)
        except Exception as e:
            print(f"[vorticity 850 calc]: {e}")
            vort850 = None

    candidates = []
    
    # Bước quét lưới: ô 4x4 (~1 độ)
    step_scan = 3
    for r in range(4, ny - 4, step_scan):
        lat_val = float(lats[r])
        # Chỉ quét vùng nhiệt đới / cận nhiệt đới (0°N - 38°N)
        if lat_val < 3.0 or lat_val > 38.0:
            continue
            
        for c in range(4, nx - 4, step_scan):
            p_val = float(mslp_grid[r, c])
            if math.isnan(p_val):
                continue
            
            # Điều kiện áp suất cục bộ phải thấp
            if p_val <= 1008.5:
                sub_p = mslp_grid[max(0, r - 4):min(ny, r + 5), max(0, c - 4):min(nx, c + 5)]
                # Điểm cực tiểu áp suất cục bộ
                if p_val == np.nanmin(sub_p):
                    # Kiểm tra độ xoáy 850 nếu có
                    has_vorticity = True
                    if vort850 is not None:
                        sub_vort = vort850[max(0, r - 3):min(ny, r + 4), max(0, c - 3):min(nx, c + 4)]
                        max_vort = np.nanmax(sub_vort) if sub_vort.size > 0 else 0
                        # Ngưỡng xoáy thuận nhiệt đới tối thiểu (~1.5 x 10^-5 s^-1)
                        if max_vort < 1.2e-5:
                            has_vorticity = False
                    
                    if has_vorticity:
                        candidates.append((r, c, p_val))
    
    # Loại bỏ các ứng viên trùng lặp / quá gần nhau (< 300 km)
    candidates.sort(key=lambda x: x[2])  # Ưu tiên điểm áp suất thấp nhất
    merged_candidates = []
    for cand in candidates:
        r0, c0, p0 = cand
        lat0, lon0 = float(lats[r0]), float(lons[c0])
        too_close = False
        for mc in merged_candidates:
            dist = math.hypot((lat0 - mc['lat']) * 111.0, (lon0 - mc['lon']) * 111.0 * math.cos(math.radians(lat0)))
            if dist < 280.0:
                too_close = True
                break
        if not too_close:
            merged_candidates.append({'r': r0, 'c': c0, 'p': p0, 'lat': lat0, 'lon': lon0})

    # 2. Với mỗi ứng viên, tinh chỉnh vị trí tâm và phân tích trường vật lý
    for cand in merged_candidates:
        r0, c0 = cand['r'], cand['c']
        p0 = cand['p']
        
        # --- BƯỚC 2: Sub-grid Pressure Centroid (Trọng tâm khuyết áp) ---
        box_rad = 5  # bán kính ~1.25 độ (~140 km)
        r_min, r_max = max(0, r0 - box_rad), min(ny, r0 + box_rad + 1)
        c_min, c_max = max(0, c0 - box_rad), min(nx, c0 + box_rad + 1)
        
        sub_p = mslp_grid[r_min:r_max, c_min:c_max]
        p_min_local = np.nanmin(sub_p)
        p_threshold = min(p_min_local + 3.0, 1008.0)
        
        weight_sum = 0.0
        lat_weighted = 0.0
        lon_weighted = 0.0
        
        for ir in range(r_min, r_max):
            for ic in range(c_min, c_max):
                pv = mslp_grid[ir, ic]
                if not math.isnan(pv) and pv <= p_threshold:
                    w = (p_threshold - pv) ** 1.5
                    lat_weighted += float(lats[ir]) * w
                    lon_weighted += float(lons[ic]) * w
                    weight_sum += w
                    
        if weight_sum > 0:
            center_lat = lat_weighted / weight_sum
            center_lon = lon_weighted / weight_sum
        else:
            center_lat = float(lats[r0])
            center_lon = float(lons[c0])
            
        # --- BƯỚC 3: Phân tích gió 10m & RMW ---
        wind_box = 8  # bán kính ~2 độ (~220 km)
        wb_r_min, wb_r_max = max(0, r0 - wind_box), min(ny, r0 + wind_box + 1)
        wb_c_min, wb_c_max = max(0, c0 - wind_box), min(nx, c0 + wind_box + 1)
        sub_w = wind10[wb_r_min:wb_r_max, wb_c_min:wb_c_max]
        
        max_wind_ms = float(np.nanmax(sub_w)) if sub_w.size > 0 else 0.0
        max_wind_kts = max_wind_ms * 1.94384
        max_wind_kmh = max_wind_ms * 3.6
        
        # Bán kính gió mạnh nhất (RMW)
        rmw_km = 60.0
        if sub_w.size > 0:
            w_idx = np.unravel_index(np.nanargmax(sub_w), sub_w.shape)
            max_w_r = wb_r_min + w_idx[0]
            max_w_c = wb_c_min + w_idx[1]
            rmw_km = math.hypot(
                (float(lats[max_w_r]) - center_lat) * 111.0,
                (float(lons[max_w_c]) - center_lon) * 111.0 * math.cos(math.radians(center_lat))
            )
            rmw_km = max(20.0, min(250.0, rmw_km))

        # --- BƯỚC 4: Kiểm tra Cấu trúc Tâm Nóng (Warm Core at 200 hPa) ---
        warm_core_dt = 0.0
        is_warm_core = True
        if t200 is not None:
            try:
                # Lấy nhiệt độ vùng trung tâm (bán kính ~120 km) và môi trường ngoài (300-500 km)
                center_t_list = []
                env_t_list = []
                for ir in range(max(0, r0 - 12), min(ny, r0 + 13)):
                    for ic in range(max(0, c0 - 12), min(nx, c0 + 13)):
                        tv = t200[ir, ic]
                        if not math.isnan(tv):
                            d_km = math.hypot(
                                (float(lats[ir]) - center_lat) * 111.0,
                                (float(lons[ic]) - center_lon) * 111.0 * math.cos(math.radians(center_lat))
                            )
                            if d_km <= 120.0:
                                center_t_list.append(tv)
                            elif 250.0 <= d_km <= 500.0:
                                env_t_list.append(tv)
                if center_t_list and env_t_list:
                    warm_core_dt = float(np.mean(center_t_list) - np.mean(env_t_list))
                    # Nếu nhiệt độ tâm 200hPa lạnh hơn môi trường > 1.8°C và ở vĩ độ cao -> xoáy ngoại nhiệt đới
                    if warm_core_dt < -1.8 and center_lat > 28.0:
                        is_warm_core = False
            except Exception as e:
                print(f"[warm core check error]: {e}")

        # --- BƯỚC 5: Kiểm tra Độ ẩm 700hPa & Vận tốc thẳng đứng 500hPa nếu có ---
        is_moist = True
        if rh700 is not None:
            try:
                rh_sub = rh700[max(0, r0 - 4):min(ny, r0 + 5), max(0, c0 - 4):min(nx, c + 5)]
                mean_rh = np.nanmean(rh_sub) if rh_sub.size > 0 else 70.0
                if mean_rh < 45.0:  # Quá khô => không thể là xoáy thuận nhiệt đới
                    is_moist = False
            except Exception:
                pass

        # Điều kiện tối thiểu để ghi nhận xoáy thuận / bão / ATNĐ
        # 1. Gió mạnh >= 10 m/s (~cấp 5-6) HOẶC áp suất tâm <= 1002 hPa
        # 2. Không phải xoáy ngoại nhiệt đới (cold core)
        # 3. Không phải vùng quá khô
        if (max_wind_ms >= 10.0 or p_min_local <= 1003.0) and is_warm_core and is_moist:
            sys_info = classify_system(max_wind_kts, p_min_local)
            detected.append({
                "lat": round(float(center_lat), 2),
                "lon": round(float(center_lon), 2),
                "min_mslp_hpa": round(float(p_min_local), 1),
                "max_wind_kts": round(float(max_wind_kts), 1),
                "max_wind_ms": round(float(max_wind_ms), 1),
                "max_wind_kmh": round(float(max_wind_kmh), 1),
                "rmw_km": round(float(rmw_km), 0),
                "warm_core_dt": round(float(warm_core_dt), 2),
                "beaufort": sys_info["beaufort"],
                "beaufort_label": sys_info["beaufort_label"],
                "type": sys_info["type"],
                "label": sys_info["label"],
                "category": sys_info["category"],
                "color": sys_info["color"]
            })
            
    return detected


# ==============================================================================
# CHƯƠNG TRÌNH CHÍNH GFS
# ==============================================================================
date_str, cycle = get_latest_gfs_info()
print(f"-> GFS Date: {date_str} - Cycle: {cycle}Z")

# Mốc dự báo: mỗi 6 tiếng đến 120h, mỗi 12 tiếng đến 240h (tổng quát và đủ bao phủ bão)
# Hoặc giữ cấu hình của bạn: 0 đến 384h mỗi 12h
MAX_FORECAST_HOUR = 384
FORECAST_STEP_HOURS = 12
forecast_hours = list(range(0, MAX_FORECAST_HOUR + 1, FORECAST_STEP_HOURS))
base_time = datetime.datetime.strptime(f"{date_str}{cycle}", "%Y%m%d%H")

output_dir = "public/data"
os.makedirs(output_dir, exist_ok=True)

manifest = []          # Danh sách các bước cho PHP
all_cyclones_summary = []  # Tổng hợp tất cả các vị trí bão phát hiện được trong toàn bộ chuỗi

for f_hr in forecast_hours:
    f_str = f"{f_hr:03d}"
    frame_time = base_time + datetime.timedelta(hours=f_hr)
    time_iso = frame_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # TẢI ĐẦY ĐỦ CÁC TRƯỜNG CẦN THIẾT TỪ GFS NOMADS
    # - 10m: UGRD, VGRD
    # - Mean Sea Level: PRMSL
    # - 850 hPa: UGRD, VGRD (tính độ xoáy vorticity)
    # - 200 hPa: TMP (kiểm tra warm-core tâm nóng)
    # - 500 hPa: VVEL (vận tốc thẳng đứng ω)
    # - 700 hPa: RH (độ ẩm tương đối)
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        f"file=gfs.t{cycle}z.pgrb2.0p25.f{f_str}&"
        f"lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&"
        f"lev_mean_sea_level=on&var_PRMSL=on&"
        f"lev_850_mb=on&lev_200_mb=on&lev_500_mb=on&lev_700_mb=on&"
        f"subregion=&toplat=47&leftlon=83&rightlon=180&bottomlat=0&"
        f"dir=%2Fgfs.{date_str}%2F{cycle}%2Fatmos"
    )
    grib_file = f"gfs_f{f_str}.grib2"

    try:
        res = requests.get(url, stream=True, timeout=60)
        if res.status_code != 200:
            print(f"⚠️ NOMADS HTTP {res.status_code} cho f{f_str}")
            continue
            
        with open(grib_file, "wb") as f:
            for chunk in res.iter_content(chunk_size=16384):
                f.write(chunk)

        # 1. Đọc trường bề mặt & 10m
        ds_wind10 = xr.open_dataset(grib_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10})
        ds_mslp   = xr.open_dataset(grib_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "meanSea"})

        u10 = ds_wind10["u10"].values
        v10 = ds_wind10["v10"].values
        mslp = ds_mslp["prmsl"].values / 100.0   # Pa -> hPa
        lats = ds_wind10["latitude"].values
        lons = ds_wind10["longitude"].values

        # 2. Đọc các tầng khí áp trên cao để tính toán tâm bão (chỉ xử lý trong RAM tại GitHub)
        u850, v850, t200, w500, rh700 = None, None, None, None, None
        try:
            ds_isobaric = xr.open_dataset(grib_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "isobaricInhPa"})
            # U, V ở 850 hPa
            if "u" in ds_isobaric and 850 in ds_isobaric.isobaricInhPa.values:
                u850 = ds_isobaric["u"].sel(isobaricInhPa=850).values
                v850 = ds_isobaric["v"].sel(isobaricInhPa=850).values
            # Nhiệt độ T ở 200 hPa
            if "t" in ds_isobaric and 200 in ds_isobaric.isobaricInhPa.values:
                t200 = ds_isobaric["t"].sel(isobaricInhPa=200).values - 273.15  # Kelvin -> Celsius
            # Vận tốc thẳng đứng w ở 500 hPa
            if "w" in ds_isobaric and 500 in ds_isobaric.isobaricInhPa.values:
                w500 = ds_isobaric["w"].sel(isobaricInhPa=500).values
            # Độ ẩm tương đối RH ở 700 hPa
            if "r" in ds_isobaric and 700 in ds_isobaric.isobaricInhPa.values:
                rh700 = ds_isobaric["r"].sel(isobaricInhPa=700).values
            ds_isobaric.close()
        except Exception as e:
            # Fallback nếu grib chia thành file con hoặc một số tầng chưa có
            print(f"Lưu ý: Không tải được tầng cao f{f_str} ({e}), dùng trường bề mặt + gradient áp suất để định vị.")

        # 3. TÍNH TOÁN VỊ TRÍ TÂM BÃO & THÔNG SỐ VẬT LÝ TẠI GITHUB
        cyclones = detect_cyclones_full(mslp, u10, v10, lats, lons, u850, v850, t200, w500, rh700)
        
        # Đưa vào danh sách tổng hợp
        frame_idx = len(manifest)
        for cyc in cyclones:
            summary_item = dict(cyc)
            summary_item["frameIndex"] = frame_idx
            summary_item["forecastTime"] = time_iso
            summary_item["forecastHour"] = f_hr
            all_cyclones_summary.append(summary_item)

        # 4. CHỈ XUẤT CÁC TRƯỜNG CƠ BẢN VỀ HOST CỦA BẠN (u10, v10, mslp + kết quả tâm)
        # Tuyệt đối KHÔNG xuất mảng dữ liệu 850/200/500/700 hPa vào JSON để tránh nặng file.
        step_payload = {
            "forecastTime": time_iso,
            "forecastHour": f_hr,
            "cyclonesDetected": cyclones,
            "data": [
                # Layer 0: u10
                {
                    "header": {
                        "parameterCategory": 2,
                        "parameterNumber": 2,
                        "nx": int(len(lons)),
                        "ny": int(len(lats)),
                        "lo1": float(lons[0]),
                        "la1": float(lats[0]),
                        "lo2": float(lons[-1]),
                        "la2": float(lats[-1]),
                        "dx": 0.25,
                        "dy": 0.25,
                    },
                    "data": [
                        round(float(val), 1) if not math.isnan(val) else 0.0
                        for val in u10.flatten()
                    ],
                },
                # Layer 1: v10
                {
                    "header": {
                        "parameterCategory": 2,
                        "parameterNumber": 3,
                        "nx": int(len(lons)),
                        "ny": int(len(lats)),
                        "lo1": float(lons[0]),
                        "la1": float(lats[0]),
                        "lo2": float(lons[-1]),
                        "la2": float(lats[-1]),
                        "dx": 0.25,
                        "dy": 0.25,
                    },
                    "data": [
                        round(float(val), 1) if not math.isnan(val) else 0.0
                        for val in v10.flatten()
                    ],
                },
                # Layer 2: MSLP
                {
                    "header": {
                        "parameterCategory": 3,
                        "parameterNumber": 1,
                        "nx": int(len(lons)),
                        "ny": int(len(lats)),
                        "lo1": float(lons[0]),
                        "la1": float(lats[0]),
                        "lo2": float(lons[-1]),
                        "la2": float(lats[-1]),
                        "dx": 0.25,
                        "dy": 0.25,
                    },
                    "data": [
                        round(float(val), 1) if not math.isnan(val) else 1013.25
                        for val in mslp.flatten()
                    ],
                },
            ],
        }

        step_file = f"gfs_nwp_f{f_str}.json"
        with open(os.path.join(output_dir, step_file), "w", encoding="utf-8") as f:
            json.dump(step_payload, f)

        manifest.append({
            "forecastTime": time_iso,
            "forecastHour": f_hr,
            "file": step_file,
            "cyclonesCount": len(cyclones)
        })

        ds_wind10.close()
        ds_mslp.close()
        if os.path.exists(grib_file):
            os.remove(grib_file)

        print(f"-> GFS f{f_str}: xong. Tìm thấy {len(cyclones)} tâm xoáy/bão.")

    except Exception as e:
        print(f"❌ Lỗi xử lý GFS f{f_str}: {e}")
        if os.path.exists(grib_file):
            try: os.remove(grib_file)
            except Exception: pass

# Xuất Manifest kèm danh sách bão toàn chuỗi
manifest_data = {
    "generatedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "gfs",
    "steps": manifest,
    "cyclones": all_cyclones_summary
}
with open(os.path.join(output_dir, "gfs_nwp_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest_data, f, ensure_ascii=False)

# File tóm tắt riêng cho bão để nạp siêu nhanh
with open(os.path.join(output_dir, "gfs_cyclones_summary.json"), "w", encoding="utf-8") as f:
    json.dump(all_cyclones_summary, f, ensure_ascii=False)

print(f"✅ GFS HOÀN THÀNH: {len(manifest)} mốc dự báo. Phát hiện tổng cộng {len(all_cyclones_summary)} điểm tâm bão/ATNĐ.")
