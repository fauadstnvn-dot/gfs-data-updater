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

    Dùng np.gradient theo tọa độ VẬT LÝ (mét, giữ đúng dấu của mảng lats/lons gốc)
    thay vì sai phân theo chỉ số lưới + hệ số "lat_sign" thủ công như trước.
    Nhờ vậy kết quả luôn đúng dấu bất kể mảng lats tăng dần (Nam -> Bắc) hay
    giảm dần (Bắc -> Nam, kiểu xuất dữ liệu mặc định của GFS/NOMADS), tránh lỗi
    đảo dấu độ xoáy đã phát hiện trước đây.
    """
    ny, nx = u_grid.shape
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)

    y_m = lats * 111195.0  # tọa độ y (m), giữ nguyên chiều/dấu của lats gốc

    dv_dx = np.full((ny, nx), np.nan, dtype=float)
    du_dy = np.full((ny, nx), np.nan, dtype=float)

    if nx > 1:
        for i in range(ny):
            lat_rad = math.radians(lats[i])
            x_m = lons * 111195.0 * max(0.05, math.cos(lat_rad))
            dv_dx[i, :] = np.gradient(v_grid[i, :], x_m)

    if ny > 1:
        for j in range(nx):
            du_dy[:, j] = np.gradient(u_grid[:, j], y_m)

    return dv_dx - du_dy


def check_closed_circulation(u_grid, v_grid, lats, lons, r_idx, c_idx,
                              radii_km=(50.0, 100.0, 150.0), n_angles=16,
                              coverage_ratio=0.7, min_speed_ms=0.5, min_tangential_ms=0.3):
    """
    Xác nhận hoàn lưu gió khép kín 360° (ngược chiều kim đồng hồ - chuẩn xoáy
    thuận Bắc bán cầu) quanh điểm ứng viên (r_idx, c_idx).

    Thay cho cách kiểm tra cứng 4 hướng cố định ở 1 bán kính duy nhất (dễ bỏ sót
    bão mini hoặc bão có mắt rộng), hàm này quét NHIỀU bán kính (50/100/150 km)
    và NHIỀU góc (mặc định 16 hướng, mỗi 22.5°) quanh tâm. Tại mỗi điểm mẫu, tính
    thành phần gió tiếp tuyến theo chiều xoáy thuận; một bán kính được coi là
    "khép kín" nếu tỷ lệ điểm có gió quay thuận đạt >= coverage_ratio. Yêu cầu đa
    số các bán kính khả dụng (>= 2/3) đều khép kín mới xác nhận là xoáy thật,
    giúp loại rãnh áp thấp / nhiễu gió mùa kéo dài một chiều.
    """
    ny, nx = u_grid.shape
    center_lat = float(lats[r_idx])

    lat_step_signed = float(lats[1] - lats[0]) if ny > 1 else 0.25
    lon_step_signed = float(lons[1] - lons[0]) if nx > 1 else 0.25
    if lat_step_signed == 0:
        lat_step_signed = 0.25
    if lon_step_signed == 0:
        lon_step_signed = 0.25

    cos_lat = max(0.2, math.cos(math.radians(center_lat)))
    closed_radius_count = 0
    tested_radius_count = 0

    for radius_km in radii_km:
        cyclonic_hits = 0
        valid_samples = 0
        for k in range(n_angles):
            phi = 2.0 * math.pi * k / n_angles  # góc toán học: 0 = Đông, tăng ngược kim đồng hồ
            dx_km = radius_km * math.cos(phi)
            dy_km = radius_km * math.sin(phi)
            dlat = dy_km / 111.0
            dlon = dx_km / (111.0 * cos_lat)

            r_off = int(round(dlat / lat_step_signed))
            c_off = int(round(dlon / lon_step_signed))
            r_s = r_idx + r_off
            c_s = c_idx + c_off
            if r_s < 0 or r_s >= ny or c_s < 0 or c_s >= nx:
                continue

            u_s = float(u_grid[r_s, c_s])
            v_s = float(v_grid[r_s, c_s])
            if math.isnan(u_s) or math.isnan(v_s):
                continue

            wind_speed = math.hypot(u_s, v_s)
            if wind_speed < min_speed_ms:
                continue  # gió quá yếu, không đủ tin cậy để đánh giá hướng quay

            valid_samples += 1
            # Thành phần gió tiếp tuyến ngược kim đồng hồ tại góc phi (u=Đông, v=Bắc)
            tangential = -u_s * math.sin(phi) + v_s * math.cos(phi)
            if tangential > min_tangential_ms:
                cyclonic_hits += 1

        if valid_samples >= max(4, n_angles // 2):
            tested_radius_count += 1
            if (cyclonic_hits / valid_samples) >= coverage_ratio:
                closed_radius_count += 1

    if tested_radius_count == 0:
        return False

    return closed_radius_count >= max(2, int(math.ceil(tested_radius_count * 2.0 / 3.0)))


def find_local_wind_lull(wind_speed_grid, lats, lons, center_r, center_c, radius_km=80.0):
    """
    Tìm điểm "lặng gió" (10m wind speed local minimum) trong bán kính radius_km
    quanh (center_r, center_c) - đặc trưng vật lý của mắt bão rõ nét (Wind Lull
    Center). Trả về (r, c) của điểm lặng gió, hoặc None nếu không xác định được.
    """
    ny, nx = wind_speed_grid.shape
    center_lat = float(lats[center_r])
    lat_deg = radius_km / 111.0
    lon_deg = radius_km / (111.0 * max(0.2, math.cos(math.radians(center_lat))))

    lat_grid_step = abs(float(lats[1] - lats[0])) if wind_speed_grid.shape[0] > 1 else 0.25
    lon_grid_step = abs(float(lons[1] - lons[0])) if wind_speed_grid.shape[1] > 1 else 0.25

    step_lat = max(1, int(round(lat_deg / lat_grid_step)))
    step_lon = max(1, int(round(lon_deg / lon_grid_step)))

    r_min, r_max = max(0, center_r - step_lat), min(ny, center_r + step_lat + 1)
    c_min, c_max = max(0, center_c - step_lon), min(nx, center_c + step_lon + 1)
    sub = wind_speed_grid[r_min:r_max, c_min:c_max]
    if sub.size == 0 or np.all(np.isnan(sub)):
        return None
    idx = np.unravel_index(np.nanargmin(sub), sub.shape)
    return (r_min + idx[0], c_min + idx[1])


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
    elif max_wind_kt >= 22:
        # Chỉ dựa vào sức gió duy trì tối đa để phân loại ATNĐ (chuẩn khí tượng),
        # KHÔNG dùng áp suất min_mslp - áp thấp nóng trên đất liền mùa hè cũng có
        # thể xuống dưới 1004 hPa dù gió chỉ 2-5 m/s.
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


def is_deep_inland(lat, lon):
    """
    Bộ lọc hình học đơn giản (không dùng land/sea mask đầy đủ) để loại các tâm
    nằm quá sâu trong lục địa châu Á - nơi xoáy thuận nhiệt đới không thể hình
    thành/duy trì (thường chỉ là "áp thấp nóng" - heat low bị nhận nhầm):
      - Nam Á / Ấn Độ, xa biển (68-88°E, 8-35°N)
      - Cao nguyên Tây Tạng & nội địa Trung Quốc, xa biển (78-105°E, >= 20°N)
    """
    if 8.0 <= lat <= 35.0 and 68.0 <= lon <= 88.0:
        return True
    if lat >= 20.0 and 78.0 <= lon <= 105.0:
        return True
    return False


def detect_cyclones_full(mslp_grid, u10_grid, v10_grid, lats, lons,
                         u850=None, v850=None, t200=None, t300=None, t250=None,
                         w500=None, rh700=None):
    """
    Thuật toán định vị tâm xoáy thuận nhiệt đới đa tầng (Multi-criteria Center Detection):

      LỚP 1 (First Guess): Cực tiểu áp suất mực biển (MSLP) quét TOÀN BỘ lưới
             0.25° (không bỏ bước lớn) để không bỏ sót mắt bão nhỏ/bão mới hình thành.
      LỚP 2 (Dynamics): Cực đại độ xoáy tương đối ζ_850 (> 2.0e-5 s^-1) quanh
             ứng viên - xác nhận có hoàn lưu xoáy thuận thật ở tầng thấp.
      LỚP 3 (Structure): Điểm "lặng gió" 10m (Wind Lull) trong bán kính 80km
             tính từ tâm xoáy 850hPa - đặc trưng mắt bão.
      LỚP 4 (Kinematics): Hoàn lưu khép kín 360° (check_closed_circulation, đa
             bán kính 50/100/150 km).
      LỚP 5 (Thermodynamics): Tâm Nóng (Warm Core) ở 300/250/200 hPa & độ ẩm
             700hPa (nếu có) để loại xoáy lạnh ngoại nhiệt đới / xoáy khô.

    Vị trí tâm cuối cùng = trung bình có trọng số 60% Trọng tâm khuyết áp
    (Pressure Centroid, box thu nhỏ ~0.75°) + 40% Tâm lặng gió/xoáy cực đại,
    tránh bị kéo lệch sang rãnh áp thấp/gió mùa rộng.
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

    # LỚP 1: Quét TOÀN BỘ lưới 0.25° (không bỏ bước), tránh bỏ sót mắt bão nhỏ
    for r in range(3, ny - 3):
        lat_val = float(lats[r])
        # Chỉ quét vùng nhiệt đới / cận nhiệt đới (0°N - 38°N)
        if lat_val < 3.0 or lat_val > 38.0:
            continue

        for c in range(3, nx - 3):
            p_val = float(mslp_grid[r, c])
            if math.isnan(p_val) or p_val > 1008.5:
                continue

            sub_p = mslp_grid[max(0, r - 3):min(ny, r + 4), max(0, c - 3):min(nx, c + 4)]
            if p_val != np.nanmin(sub_p):
                continue  # không phải cực tiểu áp suất cục bộ

            # LỚP 2: Cực đại độ xoáy 850hPa quanh ứng viên
            r_vort, c_vort = r, c
            if vort850 is not None:
                sub_vort = vort850[max(0, r - 3):min(ny, r + 4), max(0, c - 3):min(nx, c + 4)]
                if sub_vort.size == 0 or np.all(np.isnan(sub_vort)):
                    continue
                max_vort = float(np.nanmax(sub_vort))
                # Ngưỡng xoáy thuận nhiệt đới tối thiểu chuẩn WMO (~2.0 x 10^-5 s^-1)
                if max_vort < 2.0e-5:
                    continue
                v_idx = np.unravel_index(np.nanargmax(sub_vort), sub_vort.shape)
                r_vort = max(0, r - 3) + v_idx[0]
                c_vort = max(0, c - 3) + v_idx[1]

            # LỚP 4: Hoàn lưu khép kín 360° (đa bán kính)
            if not check_closed_circulation(u10_grid, v10_grid, lats, lons, r, c):
                continue

            candidates.append((r, c, p_val, r_vort, c_vort))
    
    # Loại bỏ các ứng viên trùng lặp / quá gần nhau (< 280 km)
    candidates.sort(key=lambda x: x[2])  # Ưu tiên điểm áp suất thấp nhất
    merged_candidates = []
    for cand in candidates:
        r0, c0, p0, r_vort, c_vort = cand
        lat0, lon0 = float(lats[r0]), float(lons[c0])
        too_close = False
        for mc in merged_candidates:
            dist = math.hypot((lat0 - mc['lat']) * 111.0, (lon0 - mc['lon']) * 111.0 * math.cos(math.radians(lat0)))
            if dist < 280.0:
                too_close = True
                break
        if not too_close:
            merged_candidates.append({'r': r0, 'c': c0, 'p': p0, 'lat': lat0, 'lon': lon0,
                                       'r_vort': r_vort, 'c_vort': c_vort})

    # 2. Với mỗi ứng viên, tinh chỉnh vị trí tâm và phân tích trường vật lý
    for cand in merged_candidates:
        r0, c0 = cand['r'], cand['c']
        r_vort, c_vort = cand['r_vort'], cand['c_vort']
        
        # --- Trọng tâm khuyết áp (Pressure Centroid), box thu nhỏ ~0.75° ---
        box_rad = 3  # bán kính ~0.75 độ (~85 km)
        r_min, r_max = max(0, r0 - box_rad), min(ny, r0 + box_rad + 1)
        c_min, c_max = max(0, c0 - box_rad), min(nx, c0 + box_rad + 1)
        
        sub_p = mslp_grid[r_min:r_max, c_min:c_max]
        p_min_local = float(np.nanmin(sub_p))
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
            centroid_lat = lat_weighted / weight_sum
            centroid_lon = lon_weighted / weight_sum
        else:
            centroid_lat = float(lats[r0])
            centroid_lon = float(lons[c0])

        # --- LỚP 3: Tâm lặng gió (Wind Lull) trong bán kính 80km từ tâm xoáy 850hPa ---
        structure_lat, structure_lon = float(lats[r_vort]), float(lons[c_vort])
        lull = find_local_wind_lull(wind10, lats, lons, r_vort, c_vort, radius_km=80.0)
        if lull is not None:
            lr, lc = lull
            structure_lat, structure_lon = float(lats[lr]), float(lons[lc])

        # --- Tâm cuối cùng: 60% Trọng tâm khuyết áp + 40% Tâm lặng gió/xoáy cực đại ---
        center_lat = 0.6 * centroid_lat + 0.4 * structure_lat
        center_lon = 0.6 * centroid_lon + 0.4 * structure_lon
            
        # --- Phân tích gió 10m & RMW ---
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

        # --- LỚP 5: Kiểm tra Tâm Nóng (Warm Core) ở nhiều tầng cao 300/250/200 hPa ---
        warm_core_dt = 0.0
        warm_core_layers_checked = 0
        warm_core_layers_positive = 0
        is_warm_core = True
        for t_field in (t300, t250, t200):
            if t_field is None:
                continue
            try:
                # Lấy nhiệt độ vùng trung tâm (bán kính ~120 km) và môi trường ngoài (250-500 km)
                center_t_list = []
                env_t_list = []
                for ir in range(max(0, r0 - 12), min(ny, r0 + 13)):
                    for ic in range(max(0, c0 - 12), min(nx, c0 + 13)):
                        tv = t_field[ir, ic]
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
                    dt = float(np.mean(center_t_list) - np.mean(env_t_list))
                    warm_core_layers_checked += 1
                    warm_core_dt = dt if warm_core_layers_checked == 1 else max(warm_core_dt, dt)
                    if dt > 0:
                        warm_core_layers_positive += 1
            except Exception as e:
                print(f"[warm core check error]: {e}")

        # Loại xoáy hoàn toàn lạnh ở TẤT CẢ các tầng kiểm tra được (ngoại nhiệt đới)
        if warm_core_layers_checked > 0 and warm_core_layers_positive == 0 and center_lat > 25.0:
            is_warm_core = False

        # Không có độ xoáy dương ở 850hPa tại tâm -> không đạt chuẩn xoáy thuận nhiệt đới
        if vort850 is not None:
            sub_vort_center = vort850[max(0, r0 - 3):min(ny, r0 + 4), max(0, c0 - 3):min(nx, c0 + 4)]
            if sub_vort_center.size > 0 and np.nanmax(sub_vort_center) <= 0:
                is_warm_core = False

        # --- Kiểm tra Độ ẩm 700hPa nếu có (lọc xoáy khô) ---
        is_moist = True
        if rh700 is not None:
            try:
                rh_sub = rh700[max(0, r0 - 4):min(ny, r0 + 5), max(0, c0 - 4):min(nx, c0 + 5)]
                mean_rh = np.nanmean(rh_sub) if rh_sub.size > 0 else 70.0
                if mean_rh < 45.0:  # Quá khô => không thể là xoáy thuận nhiệt đới
                    is_moist = False
            except Exception:
                pass

        # Điều kiện tối thiểu để ghi nhận xoáy thuận / bão / ATNĐ
        # 1. Gió mạnh >= 10 m/s (~20 kt) BẮT BUỘC - áp suất thấp không còn đủ để
        #    "vượt cửa" một mình (bỏ toán tử `or` cũ khiến heat low trên đất
        #    liền bị nhận nhầm thành ATNĐ khi áp suất tụt dưới 1003 hPa)
        # 2. Không phải xoáy ngoại nhiệt đới (cold core) và có xoáy dương 850hPa
        # 3. Không phải vùng quá khô
        # 4. Không nằm quá sâu trong lục địa châu Á (Ấn Độ, Tây Tạng, nội địa
        #    Trung Quốc) - nơi xoáy thuận nhiệt đới không thể hình thành/duy trì
        if (max_wind_ms >= 10.0 and p_min_local <= 1008.0 and is_warm_core and is_moist
                and not is_deep_inland(center_lat, center_lon)):
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
    # - 300/250/200 hPa: TMP (kiểm tra Warm Core đa tầng)
    # - 500 hPa: VVEL (vận tốc thẳng đứng ω)
    # - 700 hPa: RH (độ ẩm tương đối)
    # LƯU Ý: bản trước thiếu var_TMP/var_VVEL/var_RH nên T/W/RH luôn None dù có khai lev_*.
    url = (
        f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?"
        f"file=gfs.t{cycle}z.pgrb2.0p25.f{f_str}&"
        f"lev_10_m_above_ground=on&var_UGRD=on&var_VGRD=on&"
        f"lev_mean_sea_level=on&var_PRMSL=on&"
        f"lev_850_mb=on&lev_300_mb=on&lev_250_mb=on&lev_200_mb=on&lev_500_mb=on&lev_700_mb=on&"
        f"var_TMP=on&var_VVEL=on&var_RH=on&"
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
        u850, v850, t200, t300, t250, w500, rh700 = None, None, None, None, None, None, None
        try:
            ds_isobaric = xr.open_dataset(grib_file, engine="cfgrib", filter_by_keys={"typeOfLevel": "isobaricInhPa"})
            levels_available = ds_isobaric.isobaricInhPa.values
            # U, V ở 850 hPa (định vị xoáy)
            if "u" in ds_isobaric and 850 in levels_available:
                u850 = ds_isobaric["u"].sel(isobaricInhPa=850).values
                v850 = ds_isobaric["v"].sel(isobaricInhPa=850).values
            # Nhiệt độ T ở 300/250/200 hPa (kiểm tra Warm Core đa tầng)
            if "t" in ds_isobaric:
                if 300 in levels_available:
                    t300 = ds_isobaric["t"].sel(isobaricInhPa=300).values - 273.15
                if 250 in levels_available:
                    t250 = ds_isobaric["t"].sel(isobaricInhPa=250).values - 273.15
                if 200 in levels_available:
                    t200 = ds_isobaric["t"].sel(isobaricInhPa=200).values - 273.15  # Kelvin -> Celsius
            # Vận tốc thẳng đứng w ở 500 hPa
            if "w" in ds_isobaric and 500 in levels_available:
                w500 = ds_isobaric["w"].sel(isobaricInhPa=500).values
            # Độ ẩm tương đối RH ở 700 hPa
            if "r" in ds_isobaric and 700 in levels_available:
                rh700 = ds_isobaric["r"].sel(isobaricInhPa=700).values
            ds_isobaric.close()
        except Exception as e:
            # Fallback nếu grib chia thành file con hoặc một số tầng chưa có
            print(f"Lưu ý: Không tải được tầng cao f{f_str} ({e}), dùng trường bề mặt + gradient áp suất để định vị.")

        # 3. TÍNH TOÁN VỊ TRÍ TÂM BÃO & THÔNG SỐ VẬT LÝ TẠI GITHUB
        cyclones = detect_cyclones_full(mslp, u10, v10, lats, lons, u850, v850, t200, t300, t250, w500, rh700)
        
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
