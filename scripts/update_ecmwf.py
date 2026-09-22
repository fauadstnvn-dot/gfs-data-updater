import datetime
import json
import math
import os
import sys
import numpy as np
import xarray as xr
from ecmwf.opendata import Client

# ==============================================================================
# HỆ THỐNG DỰ BÁO VÀ THEO DÕI XOÁY THUẬN NHIỆT ĐỚI / BÃO TỰ ĐỘNG (ECMWF IFS 0.25°)
# Chạy trên GitHub Actions:
#  1. Tải các trường:
#     - Bề mặt / 10m: Gió 10u, 10v, Áp suất msl
#     - Tầng cao (850hPa, 200hPa, 500hPa, 700hPa) từ ECMWF Open Data
#  2. TÍNH TOÁN TẠI GITHUB:
#     - Định vị tâm sơ bộ (First Guess) bằng độ xoáy 850hPa (Vorticity)
#     - Tinh chỉnh trọng tâm khuyết áp (Pressure Centroid Method) xuống quy mô sub-grid
#     - Tính bán kính gió mạnh nhất (RMW) & tâm lặng gió
#     - Kiểm tra cấu trúc Tâm Nóng (Warm Core at 200hPa)
#     - Phân loại cấp bão quốc tế và cấp gió Beaufort Việt Nam
#  3. CHUYỂN VỀ HOST:
#     - CHỈ chuyển các trường cơ bản (u10, v10, mslp) + kết quả vị trí tâm bão đã tính.
#     - KHÔNG chuyển mảng dữ liệu 3D của các tầng cao về host.
# ==============================================================================

def compute_relative_vorticity(u_grid, v_grid, lats, lons):
    """
    Tính độ xoáy tương đối ζ = ∂v/∂x - ∂u/∂y (s^-1) trên lưới kinh vĩ cầu.

    Dùng np.gradient theo tọa độ VẬT LÝ (mét, giữ đúng dấu của mảng lats/lons gốc)
    thay vì sai phân theo chỉ số lưới + hệ số "lat_sign" thủ công như trước.
    Nhờ vậy kết quả luôn đúng dấu bất kể mảng lats tăng dần (Nam -> Bắc) hay
    giảm dần (Bắc -> Nam, kiểu xuất dữ liệu mặc định của ECMWF), tránh lỗi
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
    """Phân loại cấp bão quốc tế & cấp gió Beaufort Việt Nam."""
    max_wind_ms = max_wind_kt / 1.94384
    max_wind_kmh = max_wind_ms * 3.6
    
    if max_wind_kmh < 1:   beaufort = 0
    elif max_wind_kmh <= 5:  beaufort = 1
    elif max_wind_kmh <= 11: beaufort = 2
    elif max_wind_kmh <= 19: beaufort = 3
    elif max_wind_kmh <= 28: beaufort = 4
    elif max_wind_kmh <= 38: beaufort = 5
    elif max_wind_kmh <= 49: beaufort = 6
    elif max_wind_kmh <= 61: beaufort = 7
    elif max_wind_kmh <= 74: beaufort = 8
    elif max_wind_kmh <= 88: beaufort = 9
    elif max_wind_kmh <= 102: beaufort = 10
    elif max_wind_kmh <= 117: beaufort = 11
    elif max_wind_kmh <= 133: beaufort = 12
    elif max_wind_kmh <= 149: beaufort = 13
    elif max_wind_kmh <= 166: beaufort = 14
    elif max_wind_kmh <= 183: beaufort = 15
    elif max_wind_kmh <= 201: beaufort = 16
    else: beaufort = 17

    beaufort_label = f"Cấp {beaufort} ({round(max_wind_kmh)} km/h)"

    if max_wind_kt >= 100:
        return {"type": "SUPER_TYPHOON", "label": "Siêu bão", "category": 5, "beaufort": beaufort, "beaufort_label": beaufort_label, "color": "#ef4444"}
    elif max_wind_kt >= 64:
        return {"type": "TYPHOON", "label": "Bão rất mạnh (Cuồng phong)", "category": 4, "beaufort": beaufort, "beaufort_label": beaufort_label, "color": "#f97316"}
    elif max_wind_kt >= 48:
        return {"type": "SEVERE_TROPICAL_STORM", "label": "Bão mạnh", "category": 3, "beaufort": beaufort, "beaufort_label": beaufort_label, "color": "#eab308"}
    elif max_wind_kt >= 34:
        return {"type": "TROPICAL_STORM", "label": "Bão nhiệt đới", "category": 2, "beaufort": beaufort, "beaufort_label": beaufort_label, "color": "#3b82f6"}
    elif max_wind_kt >= 22 or min_mslp <= 1004.0:
        return {"type": "TROPICAL_DEPRESSION", "label": "Áp thấp nhiệt đới", "category": 1, "beaufort": beaufort, "beaufort_label": beaufort_label, "color": "#06b6d4"}
    else:
        return {"type": "LOW_PRESSURE", "label": "Vùng áp thấp", "category": 0, "beaufort": beaufort, "beaufort_label": beaufort_label, "color": "#64748b"}


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
    
    vort850 = None
    if u850 is not None and v850 is not None:
        try:
            vort850 = compute_relative_vorticity(u850, v850, lats, lons)
        except Exception:
            vort850 = None

    candidates = []
    # LỚP 1: Quét TOÀN BỘ lưới 0.25° (không bỏ bước), tránh bỏ sót mắt bão nhỏ
    for r in range(3, ny - 3):
        lat_val = float(lats[r])
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

    # Loại ứng viên quá gần nhau (< 280 km), ưu tiên điểm áp suất thấp nhất
    candidates.sort(key=lambda x: x[2])
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

    # Tinh chỉnh tọa độ tâm và kiểm tra tính chất vật lý
    for cand in merged_candidates:
        r0, c0 = cand['r'], cand['c']
        r_vort, c_vort = cand['r_vort'], cand['c_vort']

        # --- Trọng tâm khuyết áp (Pressure Centroid), box thu nhỏ ~0.75° ---
        box_rad = 3
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

        # --- Gió 10m & RMW ---
        wind_box = 8
        wb_r_min, wb_r_max = max(0, r0 - wind_box), min(ny, r0 + wind_box + 1)
        wb_c_min, wb_c_max = max(0, c0 - wind_box), min(nx, c0 + wind_box + 1)
        sub_w = wind10[wb_r_min:wb_r_max, wb_c_min:wb_c_max]
        
        max_wind_ms = float(np.nanmax(sub_w)) if sub_w.size > 0 else 0.0
        max_wind_kts = max_wind_ms * 1.94384
        max_wind_kmh = max_wind_ms * 3.6
        
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
                center_t_list = []
                env_t_list = []
                for ir in range(max(0, r0 - 12), min(ny, r0 + 13)):
                    for ic in range(max(0, c0 - 12), min(nx, c0 + 13)):
                        tv = t_field[ir, ic]
                        if math.isnan(tv):
                            continue
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
            except Exception:
                pass

        # Loại xoáy hoàn toàn lạnh ở TẤT CẢ các tầng kiểm tra được (ngoại nhiệt đới)
        if warm_core_layers_checked > 0 and warm_core_layers_positive == 0 and center_lat > 25.0:
            is_warm_core = False

        # Không có độ xoáy dương ở 850hPa tại tâm -> không đạt chuẩn xoáy thuận nhiệt đới
        if vort850 is not None:
            sub_vort_center = vort850[max(0, r0 - 3):min(ny, r0 + 4), max(0, c0 - 3):min(nx, c0 + 4)]
            if sub_vort_center.size > 0 and np.nanmax(sub_vort_center) <= 0:
                is_warm_core = False

        # --- Độ ẩm 700hPa nếu có (lọc xoáy khô) ---
        is_moist = True
        if rh700 is not None:
            try:
                rh_sub = rh700[max(0, r0 - 4):min(ny, r0 + 5), max(0, c0 - 4):min(nx, c0 + 5)]
                mean_rh = float(np.nanmean(rh_sub)) if rh_sub.size > 0 else 70.0
                if mean_rh < 45.0:
                    is_moist = False
            except Exception:
                pass

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
# CHƯƠNG TRÌNH CHÍNH ECMWF
# ==============================================================================
print("-> Bắt đầu kết nối tải dữ liệu ECMWF Open Data...")
client = Client(source="ecmwf", model="ifs", resol="0p25")

# Chuỗi mốc dự báo ECMWF IFS
forecast_steps = list(range(0, 145, 6)) + list(range(150, 241, 12))
manifest = []
all_cyclones_summary = []

area_crop = [47, 83, 0, 180]
output_dir = "public/data"
os.makedirs(output_dir, exist_ok=True)

for step in forecast_steps:
    grib_surface = f"ecmwf_sfc_{step}.grib2"
    grib_upper   = f"ecmwf_pl_{step}.grib2"

    try:
        # 1. Tải bề mặt (10u, 10v, msl)
        client.retrieve(
            datetime=0,
            stream="oper",
            type="fc",
            step=step,
            param=["10u", "10v", "msl"],
            target=grib_surface,
        )

        ds_sfc = xr.open_dataset(grib_surface, engine="cfgrib")
        ds_sfc_sub = ds_sfc.sel(
            latitude=slice(area_crop[0], area_crop[2]),
            longitude=slice(area_crop[1], area_crop[3]),
        )

        lats = ds_sfc_sub["latitude"].values
        lons = ds_sfc_sub["longitude"].values
        u10 = ds_sfc_sub["u10"].values
        v10 = ds_sfc_sub["v10"].values
        mslp = ds_sfc_sub["msl"].values / 100.0  # Pa -> hPa
        valid_time = str(ds_sfc_sub["valid_time"].values)[:19] + "Z"

        # 2. Cố gắng tải các tầng cao từ ECMWF Open Data nếu có
        # 850hPa (u,v): định vị xoáy | 300/250/200hPa (t): kiểm tra Warm Core đa tầng
        u850, v850, t200, t300, t250, w500, rh700 = None, None, None, None, None, None, None
        try:
            client.retrieve(
                datetime=0,
                stream="oper",
                type="fc",
                step=step,
                param=["u", "v", "t"],
                levelist=[850, 300, 250, 200],
                target=grib_upper,
            )
            ds_pl = xr.open_dataset(grib_upper, engine="cfgrib")
            ds_pl_sub = ds_pl.sel(
                latitude=slice(area_crop[0], area_crop[2]),
                longitude=slice(area_crop[1], area_crop[3]),
            )
            levels_available = ds_pl_sub.isobaricInhPa.values
            if "u" in ds_pl_sub and 850 in levels_available:
                u850 = ds_pl_sub["u"].sel(isobaricInhPa=850).values
                v850 = ds_pl_sub["v"].sel(isobaricInhPa=850).values
            if "t" in ds_pl_sub:
                if 300 in levels_available:
                    t300 = ds_pl_sub["t"].sel(isobaricInhPa=300).values - 273.15
                if 250 in levels_available:
                    t250 = ds_pl_sub["t"].sel(isobaricInhPa=250).values - 273.15
                if 200 in levels_available:
                    t200 = ds_pl_sub["t"].sel(isobaricInhPa=200).values - 273.15
            ds_pl.close()
            if os.path.exists(grib_upper):
                os.remove(grib_upper)
        except Exception as e:
            # Fallback nếu open data không hỗ trợ levelist pl ở gói này
            pass

        # 3. TÍNH TOÁN VỊ TRÍ TÂM BÃO TẠI GITHUB
        cyclones = detect_cyclones_full(mslp, u10, v10, lats, lons, u850, v850, t200, t300, t250, w500, rh700)

        frame_idx = len(manifest)
        for cyc in cyclones:
            summary_item = dict(cyc)
            summary_item["frameIndex"] = frame_idx
            summary_item["forecastTime"] = valid_time
            summary_item["forecastHour"] = step
            all_cyclones_summary.append(summary_item)

        # 4. CHỈ XUẤT DỮ LIỆU CẦN THIẾT VỀ MÁY CHỦ
        step_data = {
            "forecastTime": valid_time,
            "forecastHour": step,
            "cyclonesDetected": cyclones,
            "data": [
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

        step_file = f"ecmwf_nwp_f{step:03d}.json"
        with open(os.path.join(output_dir, step_file), "w", encoding="utf-8") as f:
            json.dump(step_data, f)

        manifest.append({
            "forecastTime": valid_time,
            "forecastHour": step,
            "file": step_file,
            "cyclonesCount": len(cyclones)
        })

        ds_sfc.close()
        if os.path.exists(grib_surface):
            os.remove(grib_surface)
        if os.path.exists(grib_upper):
            os.remove(grib_upper)

        print(f"-> ECMWF +{step}h: xong. Phát hiện {len(cyclones)} tâm bão/ATNĐ.")

    except Exception as e:
        print(f"❌ Lỗi mốc ECMWF step {step}: {e}")
        if os.path.exists(grib_surface):
            try: os.remove(grib_surface)
            except Exception: pass
        if os.path.exists(grib_upper):
            try: os.remove(grib_upper)
            except Exception: pass

# Xuất Manifest kèm danh sách bão
manifest_data = {
    "generatedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "model": "ecmwf",
    "steps": manifest,
    "cyclones": all_cyclones_summary
}
with open(os.path.join(output_dir, "ecmwf_nwp_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest_data, f, ensure_ascii=False)

with open(os.path.join(output_dir, "ecmwf_cyclones_summary.json"), "w", encoding="utf-8") as f:
    json.dump(all_cyclones_summary, f, ensure_ascii=False)

print(f"✅ ECMWF HOÀN THÀNH: {len(manifest)} mốc dự báo. Phát hiện {len(all_cyclones_summary)} điểm bão/ATNĐ.")
