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
    """Tính độ xoáy tương đối ζ = ∂v/∂x - ∂u/∂y (s^-1) trên lưới cầu."""
    ny, nx = u_grid.shape
    vort = np.zeros_like(u_grid, dtype=float)
    dy_deg = abs(float(lats[1] - lats[0])) if ny > 1 else 0.25
    dx_deg = abs(float(lons[1] - lons[0])) if nx > 1 else 0.25
    dy_m = dy_deg * 111195.0
    lat_sign = 1.0 if lats[-1] > lats[0] else -1.0
    
    for i in range(1, ny - 1):
        lat_rad = math.radians(float(lats[i]))
        dx_m = dx_deg * 111195.0 * max(0.05, math.cos(lat_rad))
        dv_dx = (v_grid[i, 2:] - v_grid[i, :-2]) / (2.0 * dx_m)
        du_dy = lat_sign * (u_grid[i + 1, 1:-1] - u_grid[i - 1, 1:-1]) / (2.0 * dy_m)
        vort[i, 1:-1] = dv_dx - du_dy
    return vort


def check_closed_circulation(u_grid, v_grid, lats, lons, r_idx, c_idx, radius_km=130.0):
    """
    Kiểm tra xem điểm (r_idx, c_idx) có tạo thành một vòng xoáy gió khép kín 360 độ
    (ngược chiều kim đồng hồ ở Bắc bán cầu) hay không.
    Trả về: True (là xoáy khép kín) hoặc False (chỉ là rãnh thấp/nhiễu gió).
    """
    ny, nx = u_grid.shape
    center_lat = float(lats[r_idx])

    quadrants = [False, False, False, False]

    lat_deg_rad = radius_km / 111.0
    lon_deg_rad = radius_km / (111.0 * max(0.2, math.cos(math.radians(center_lat))))

    grid_step_lat = max(1, int(round(lat_deg_rad / 0.25)))
    grid_step_lon = max(1, int(round(lon_deg_rad / 0.25)))

    lat_ascending = lats[-1] > lats[0]

    # 1. Phía BẮC: gió phải thổi từ Đông sang Tây (u < 0)
    r_north = max(0, r_idx - grid_step_lat) if not lat_ascending else min(ny - 1, r_idx + grid_step_lat)
    u_north = u_grid[r_north, max(0, c_idx - 2):min(nx, c_idx + 3)]
    if u_north.size > 0 and np.nanmean(u_north) < -1.5:
        quadrants[0] = True

    # 2. Phía ĐÔNG: gió phải thổi từ Nam lên Bắc (v > 0)
    c_east = min(nx - 1, c_idx + grid_step_lon)
    v_east = v_grid[max(0, r_idx - 2):min(ny, r_idx + 3), c_east]
    if v_east.size > 0 and np.nanmean(v_east) > 1.5:
        quadrants[1] = True

    # 3. Phía NAM: gió phải thổi từ Tây sang Đông (u > 0)
    r_south = min(ny - 1, r_idx + grid_step_lat) if not lat_ascending else max(0, r_idx - grid_step_lat)
    u_south = u_grid[r_south, max(0, c_idx - 2):min(nx, c_idx + 3)]
    if u_south.size > 0 and np.nanmean(u_south) > 1.5:
        quadrants[2] = True

    # 4. Phía TÂY: gió phải thổi từ Bắc xuống Nam (v < 0)
    c_west = max(0, c_idx - grid_step_lon)
    v_west = v_grid[max(0, r_idx - 2):min(ny, r_idx + 3), c_west]
    if v_west.size > 0 and np.nanmean(v_west) < -1.5:
        quadrants[3] = True

    # Cần ít nhất 3/4 góc phần tư xoay đúng chiều mới xác nhận là xoáy khép kín
    return sum(quadrants) >= 3


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
                         u850=None, v850=None, t200=None, w500=None, rh700=None):
    """
    Thuật toán phát hiện tâm bão chuẩn khoa học khí tượng:
    - Tìm First Guess ứng viên bằng áp suất & độ xoáy 850hPa
    - Tinh chỉnh tọa độ trọng tâm khuyết áp (Pressure Centroid Method)
    - Tính RMW, bán kính gió và kiểm tra Warm Core
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
    step_scan = 3
    for r in range(4, ny - 4, step_scan):
        lat_val = float(lats[r])
        if lat_val < 3.0 or lat_val > 38.0:
            continue
            
        for c in range(4, nx - 4, step_scan):
            p_val = float(mslp_grid[r, c])
            if math.isnan(p_val):
                continue
            
            if p_val <= 1008.5:
                sub_p = mslp_grid[max(0, r - 4):min(ny, r + 5), max(0, c - 4):min(nx, c + 5)]
                if p_val == np.nanmin(sub_p):
                    has_vorticity = True
                    if vort850 is not None:
                        sub_vort = vort850[max(0, r - 3):min(ny, r + 4), max(0, c - 3):min(nx, c + 4)]
                        max_vort = np.nanmax(sub_vort) if sub_vort.size > 0 else 0
                        if max_vort < 1.2e-5:
                            has_vorticity = False
                    if has_vorticity:
                        # Kiểm tra vòng xoáy gió khép kín để loại rãnh áp thấp / nhiễu gió mùa
                        is_closed = check_closed_circulation(u10_grid, v10_grid, lats, lons, r, c, radius_km=130.0)
                        if is_closed:
                            candidates.append((r, c, p_val))
    
    # Loại ứng viên quá gần nhau (< 280 km)
    candidates.sort(key=lambda x: x[2])
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

    # Tinh chỉnh tọa độ và kiểm tra tính chất
    for cand in merged_candidates:
        r0, c0 = cand['r'], cand['c']
        p0 = cand['p']
        
        # Trọng tâm khuyết áp (Pressure Centroid Method)
        box_rad = 5
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
            
        # Gió 10m & RMW
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

        # Kiểm tra Warm Core ở 200 hPa
        warm_core_dt = 0.0
        is_warm_core = True
        if t200 is not None:
            try:
                center_t = []
                env_t = []
                for ir in range(max(0, r0 - 12), min(ny, r0 + 13)):
                    for ic in range(max(0, c0 - 12), min(nx, c0 + 13)):
                        tv = t200[ir, ic]
                        if not math.isnan(tv):
                            d_km = math.hypot((float(lats[ir]) - center_lat) * 111.0, (float(lons[ic]) - center_lon) * 111.0 * math.cos(math.radians(center_lat)))
                            if d_km <= 120.0:
                                center_t.append(tv)
                            elif 250.0 <= d_km <= 500.0:
                                env_t.append(tv)
                if center_t and env_t:
                    warm_core_dt = float(np.mean(center_t) - np.mean(env_t))
                    if warm_core_dt < -1.8 and center_lat > 28.0:
                        is_warm_core = False
            except Exception:
                pass

        if (max_wind_ms >= 10.0 or p_min_local <= 1003.0) and is_warm_core:
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
        u850, v850, t200, w500, rh700 = None, None, None, None, None
        try:
            client.retrieve(
                datetime=0,
                stream="oper",
                type="fc",
                step=step,
                param=["u", "v", "t"],
                levelist=[850, 200],
                target=grib_upper,
            )
            ds_pl = xr.open_dataset(grib_upper, engine="cfgrib")
            ds_pl_sub = ds_pl.sel(
                latitude=slice(area_crop[0], area_crop[2]),
                longitude=slice(area_crop[1], area_crop[3]),
            )
            if "u" in ds_pl_sub and 850 in ds_pl_sub.isobaricInhPa.values:
                u850 = ds_pl_sub["u"].sel(isobaricInhPa=850).values
                v850 = ds_pl_sub["v"].sel(isobaricInhPa=850).values
            if "t" in ds_pl_sub and 200 in ds_pl_sub.isobaricInhPa.values:
                t200 = ds_pl_sub["t"].sel(isobaricInhPa=200).values - 273.15
            ds_pl.close()
            if os.path.exists(grib_upper):
                os.remove(grib_upper)
        except Exception as e:
            # Fallback nếu open data không hỗ trợ levelist pl ở gói này
            pass

        # 3. TÍNH TOÁN VỊ TRÍ TÂM BÃO TẠI GITHUB
        cyclones = detect_cyclones_full(mslp, u10, v10, lats, lons, u850, v850, t200, w500, rh700)

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
