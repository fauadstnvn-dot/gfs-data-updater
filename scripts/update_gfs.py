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


def smooth_grid(grid, passes=1):
    """
    Làm mượt nhẹ trường 2D bằng bộ lọc trung bình 3x3 (bỏ qua NaN), lặp `passes` lần.

    MỤC ĐÍCH (sửa lỗi tâm bão nằm ngoài đường đẳng áp khép kín):
      - ĐỒNG BỘ trường dùng để DÒ TÂM với trường đã làm mượt mà PHP dùng để vẽ
        đường đẳng áp. Nhờ đó tâm bão và đường đẳng áp luôn khớp nhau; không còn
        cảnh tâm nằm ở nơi trên bản đồ không hề có vòng đẳng áp khép kín.
      - Triệt các cực tiểu áp suất GIẢ quy mô 1-2 ô lưới sinh ra khi mô hình quy
        đổi (ngoại suy) áp suất mực biển xuống DƯỚI địa hình núi cao (Đài Loan,
        Luzon, Nhật...). Một "áp thấp" chỉ tồn tại ở trường thô mà biến mất sau
        khi làm mượt thì cũng biến mất khỏi bản đồ -> không phải tâm thật.
    """
    if grid is None:
        return None
    g = np.array(grid, dtype=float)
    for _ in range(max(0, passes)):
        acc = np.zeros_like(g)
        cnt = np.zeros_like(g)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                shifted = np.roll(np.roll(g, dr, axis=0), dc, axis=1)
                valid = ~np.isnan(shifted)
                acc[valid] += shifted[valid]
                cnt[valid] += 1.0
        with np.errstate(invalid="ignore"):
            g = np.where(cnt > 0, acc / cnt, g)
    return g


def build_high_terrain_mask(orog, threshold_m=300.0):
    """
    Tạo mặt nạ boolean (True = địa hình cao) từ trường độ cao địa hình orography
    (mét). Áp suất mực biển tại các ô có độ cao > threshold_m là con số NGOẠI SUY
    xuống dưới lòng đất, thường sinh cực trị áp giả -> loại khỏi vùng dò tâm.
    Trả về None nếu không có dữ liệu orography (khi đó dùng fallback is_high_terrain).
    """
    if orog is None:
        return None
    try:
        arr = np.asarray(orog, dtype=float)
        return arr > threshold_m
    except Exception:
        return None


def is_high_terrain(lat, lon):
    """
    Fallback khi KHÔNG có trường độ cao địa hình (orography): loại thủ công các
    dải núi cao ven biển Đông Á - nơi áp suất mực biển bị ngoại suy xuống dưới
    mặt đất, sinh cực tiểu áp GIẢ (không phải xoáy thuận nhiệt đới thật):
      - Dải Trung Ương Sơn (Đài Loan)
      - Dãy Cordillera (Bắc Luzon, Philippines)
      - Vùng núi Honshu (Nhật Bản)
    """
    if 22.2 <= lat <= 24.9 and 120.6 <= lon <= 121.7:   # Taiwan Central Range
        return True
    if 16.0 <= lat <= 18.6 and 120.5 <= lon <= 121.7:   # Luzon Cordillera
        return True
    if 34.5 <= lat <= 38.6 and 136.0 <= lon <= 140.6:   # Japan Alps / Honshu
        return True
    return False


def symmetric_tangential_wind(u_grid, v_grid, lats, lons, center_lat, center_lon,
                              radii_km=(50.0, 100.0, 150.0, 200.0), n_angles=24):
    """
    Tính GIÓ TIẾP TUYẾN ĐỐI XỨNG (azimuthal-mean tangential wind) quanh tâm -
    thước đo mức độ QUAY THÀNH XOÁY thực sự của hệ thống.

    Khác với cách cũ lấy `np.nanmax` gió trong một hộp vuông (dễ vồ nhầm luồng
    gió mùa/gió tăng tốc do địa hình chạy MỘT CHIỀU và gán thành cường độ bão),
    hàm này lấy TRUNG BÌNH thành phần gió tiếp tuyến theo vòng tròn quanh tâm:
      - Xoáy thật: gió quay quanh tâm -> trung bình tiếp tuyến LỚN (> 0).
      - Luồng gió thẳng: hai nửa vòng triệt tiêu nhau -> trung bình ~ 0.
    Trả về (vt_sym_max_ms, rmw_km) với vt_sym_max_ms là giá trị lớn nhất theo các
    bán kính khảo sát và rmw_km là bán kính đạt giá trị đó.
    """
    ny, nx = u_grid.shape
    lat_step = float(lats[1] - lats[0]) if ny > 1 else -0.25
    lon_step = float(lons[1] - lons[0]) if nx > 1 else 0.25
    if lat_step == 0:
        lat_step = -0.25
    if lon_step == 0:
        lon_step = 0.25
    cos_lat = max(0.2, math.cos(math.radians(center_lat)))

    best_vt = 0.0
    best_r = radii_km[0]
    for radius_km in radii_km:
        vt_sum = 0.0
        cnt = 0
        for k in range(n_angles):
            phi = 2.0 * math.pi * k / n_angles  # 0 = Đông, tăng ngược kim đồng hồ
            dlat = (radius_km * math.sin(phi)) / 111.0
            dlon = (radius_km * math.cos(phi)) / (111.0 * cos_lat)
            r_s = int(round((center_lat + dlat - float(lats[0])) / lat_step))
            c_s = int(round((center_lon + dlon - float(lons[0])) / lon_step))
            if r_s < 0 or r_s >= ny or c_s < 0 or c_s >= nx:
                continue
            u_s = float(u_grid[r_s, c_s])
            v_s = float(v_grid[r_s, c_s])
            if math.isnan(u_s) or math.isnan(v_s):
                continue
            vt = -u_s * math.sin(phi) + v_s * math.cos(phi)  # ngược kim đồng hồ
            vt_sum += vt
            cnt += 1
        if cnt >= int(n_angles * 0.6):
            vt_mean = vt_sum / cnt
            if vt_mean > best_vt:
                best_vt = vt_mean
                best_r = radius_km
    return best_vt, best_r


def check_closed_circulation(u_grid, v_grid, lats, lons, r_idx, c_idx,
                              radii_km=(50.0, 100.0), n_angles=16,  # bỏ bán kính 150km, dễ dính nhiễu ngoại vi
                              coverage_ratio=0.75, min_speed_ms=2.5, min_tangential_ms=1.5):
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

            # Lấy trung bình một cụm 3x3 ô lưới quanh điểm mẫu (thay vì 1 pixel đơn
            # lẻ) để chống nhiễu địa hình (topographic eddy) - gió giật cục bộ khi
            # luồng gió thẳng va vào núi/đồi dễ tạo ra 1-2 ô lưới có hướng "giả xoáy"
            # nhưng trung bình cụm sẽ triệt tiêu nhiễu này.
            r_lo, r_hi = max(0, r_s - 1), min(ny, r_s + 2)
            c_lo, c_hi = max(0, c_s - 1), min(nx, c_s + 2)
            u_s = float(np.nanmean(u_grid[r_lo:r_hi, c_lo:c_hi]))
            v_s = float(np.nanmean(v_grid[r_lo:r_hi, c_lo:c_hi]))
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
                         w500=None, rh700=None, orog=None):
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

    # Trường MSLP đã làm mượt: DÙNG CHO TOÀN BỘ việc dò tâm & đo độ sâu, để tâm
    # bão luôn khớp với đường đẳng áp (PHP cũng vẽ trên trường đã làm mượt) và
    # để triệt cực tiểu áp giả do địa hình núi cao.
    mslp_det = smooth_grid(mslp_grid, passes=1)
    if mslp_det is None:
        mslp_det = mslp_grid

    # Mặt nạ địa hình cao (nếu có orography); nếu không, dùng fallback theo tọa độ.
    terrain_high = build_high_terrain_mask(orog, threshold_m=300.0)

    def _on_high_terrain(la, lo, r=None, c=None):
        if terrain_high is not None:
            if r is None or c is None:
                lat_step = float(lats[1] - lats[0]) if ny > 1 else -0.25
                lon_step = float(lons[1] - lons[0]) if nx > 1 else 0.25
                if lat_step == 0:
                    lat_step = -0.25
                if lon_step == 0:
                    lon_step = 0.25
                r = int(round((la - float(lats[0])) / lat_step))
                c = int(round((lo - float(lons[0])) / lon_step))
            if 0 <= r < ny and 0 <= c < nx:
                return bool(terrain_high[r, c])
            return False
        return is_high_terrain(la, lo)

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
            # Loại ngay các ô nằm trên địa hình cao (áp mực biển bị ngoại suy)
            if _on_high_terrain(float(lats[r]), float(lons[c]), r, c):
                continue

            p_val = float(mslp_det[r, c])
            if math.isnan(p_val) or p_val > 1008.5:
                continue

            sub_p = mslp_det[max(0, r - 3):min(ny, r + 4), max(0, c - 3):min(nx, c + 4)]
            if p_val != np.nanmin(sub_p):
                continue  # không phải cực tiểu áp suất cục bộ

            # Bắt buộc tâm khuyết áp phải sâu hơn rìa box (7x7) ít nhất 2.0 hPa
            # (trên trường ĐÃ LÀM MƯỢT) - tương đương phải tồn tại một đường đẳng
            # áp khép kín thật quanh tâm. Ngưỡng 1.0 hPa cũ quá lỏng, để lọt các
            # cực tiểu áp giả nông (đặc biệt là artifact địa hình ~1 hPa).
            edge_max = float(np.nanmax(sub_p))
            if (edge_max - p_val) < 2.0:
                continue

            # LỚP 2: Cực đại độ xoáy 850hPa quanh ứng viên
            r_vort, c_vort = r, c
            if vort850 is not None:
                sub_vort = vort850[max(0, r - 3):min(ny, r + 4), max(0, c - 3):min(nx, c + 4)]
                if sub_vort.size == 0 or np.all(np.isnan(sub_vort)):
                    continue
                max_vort = float(np.nanmax(sub_vort))
                # Ngưỡng xoáy thuận nhiệt đới tối thiểu chuẩn WMO (~2.0 x 10^-5 s^-1)
                # Nâng ngưỡng từ 2.0e-5 lên 3.5e-5: áp thấp địa hình (topographic low)
                # trên đất liền có thể sinh độ xoáy cục bộ nhưng không đạt cường độ
                # của một xoáy thuận nhiệt đới thực thụ ở 850hPa.
                if max_vort < 3.5e-5:
                    continue
                v_idx = np.unravel_index(np.nanargmax(sub_vort), sub_vort.shape)
                r_vort = max(0, r - 3) + v_idx[0]
                c_vort = max(0, c - 3) + v_idx[1]

            # LỚP 4: Hoàn lưu khép kín 360° (đa bán kính)
            # Kiểm tra quanh tâm gió xoáy (r_vort, c_vort) thay vì tâm MSLP (r, c) -
            # với bão mới hình thành / bị đứt gió, tâm áp suất và tâm động lực
            # thường lệch 30-80km, quét quanh tâm MSLP dễ bỏ sót bão thật.
            if not check_closed_circulation(u10_grid, v10_grid, lats, lons, r_vort, c_vort):
                continue

            candidates.append((r, c, p_val, r_vort, c_vort))
    
    # Loại bỏ các ứng viên trùng lặp / vệ tinh nhiễu của một hệ thống mạnh hơn
    # gần đó (mesovortex trong dải mây xoắn ngoài của bão chính). Ngưỡng cố
    # định 280 km chỉ đủ diệt các tâm bị dò trùng SÁT nhau; bão mạnh có hoàn
    # lưu ngoài rộng 300-600km dư sức sinh cực tiểu áp/xoáy vệ tinh cục bộ ở xa
    # hơn nhưng vẫn không phải một xoáy thuận độc lập. Một cặp bão đôi thật
    # (Fujiwhara) thường cách xa nhau > 500-600 km và cường độ tương đương.
    # -> Giãn ngưỡng loại trùng THEO ĐỘ SÂU ÁP SUẤT của hệ mạnh hơn đã được
    # chấp nhận trước (candidates đã sort theo áp tăng dần nên mc luôn mạnh
    # hơn hoặc bằng ứng viên đang xét).
    #
    # SỬA LỖI TIẾP (điểm nhiễu vẫn lọt qua ở một số thời điểm): ngưỡng khoảng
    # cách đơn thuần vẫn có thể không đủ khi 2 tâm nằm hơi xa nhau nhưng thực
    # chất chỉ là 2 cực tiểu cục bộ (double/multiple minima) của CÙNG MỘT vùng
    # áp thấp rộng, không có một rặng áp cao (ridge/saddle) tách biệt thật giữa
    # chúng. Đây chính là định nghĩa khí tượng để phân biệt "2 xoáy thuận độc
    # lập" (có yên áp/saddle cao hơn đáng kể ngăn giữa) với "1 xoáy có nhiễu
    # cấu trúc nội tại". Bổ sung kiểm tra yên áp (saddle check): lấy giá trị
    # MSLP lớn nhất dọc đường thẳng nối 2 tâm trên lưới đã làm mượt - nếu rặng
    # áp cao đó không cao hơn tâm YẾU HƠN ít nhất SADDLE_MIN_HPA, coi 2 tâm là
    # cùng một hệ thống và loại tâm yếu hơn.
    SADDLE_MIN_HPA = 1.5

    def _saddle_max_pressure(lat0, lon0, lat1, lon1):
        lat_step = float(lats[1] - lats[0]) if ny > 1 else -0.25
        lon_step = float(lons[1] - lons[0]) if nx > 1 else 0.25
        if lat_step == 0:
            lat_step = -0.25
        if lon_step == 0:
            lon_step = 0.25
        dist_km = math.hypot((lat0 - lat1) * 111.0, (lon0 - lon1) * 111.0 * math.cos(math.radians((lat0 + lat1) / 2.0)))
        n_samples = max(2, int(dist_km / 25.0))  # lấy mẫu mỗi ~25km
        max_p = -np.inf
        for i in range(n_samples + 1):
            f = i / n_samples
            la = lat0 + (lat1 - lat0) * f
            lo = lon0 + (lon1 - lon0) * f
            r = int(round((la - float(lats[0])) / lat_step))
            c = int(round((lo - float(lons[0])) / lon_step))
            if 0 <= r < ny and 0 <= c < nx:
                v = float(mslp_det[r, c])
                if not math.isnan(v) and v > max_p:
                    max_p = v
        return max_p if max_p != -np.inf else None

    candidates.sort(key=lambda x: x[2])  # Ưu tiên điểm áp suất thấp nhất (mạnh nhất) trước
    merged_candidates = []
    for cand in candidates:
        r0, c0, p0, r_vort, c_vort = cand
        lat0, lon0 = float(lats[r0]), float(lons[c0])
        too_close = False
        for mc in merged_candidates:
            dist = math.hypot((lat0 - mc['lat']) * 111.0, (lon0 - mc['lon']) * 111.0 * math.cos(math.radians(lat0)))
            depth_mc = max(0.0, 1010.0 - mc['p'])  # độ sâu áp suất (hPa) của hệ mạnh hơn
            min_sep_km = 280.0 + min(320.0, depth_mc * 6.0)  # tối đa 600km với bão rất sâu
            if dist < min_sep_km:
                too_close = True
                break
            # Kiểm tra yên áp: nếu không có rặng áp cao thật ngăn giữa 2 tâm
            # (rặng chỉ cao hơn tâm yếu hơn p0 chưa tới SADDLE_MIN_HPA), đây chỉ
            # là nhiễu cấu trúc của cùng một vùng áp thấp -> loại tâm yếu hơn dù
            # khoảng cách đã vượt min_sep_km.
            saddle_p = _saddle_max_pressure(lat0, lon0, mc['lat'], mc['lon'])
            if saddle_p is not None and (saddle_p - p0) < SADDLE_MIN_HPA:
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
        
        sub_p = mslp_det[r_min:r_max, c_min:c_max]
        p_min_local = float(np.nanmin(sub_p))
        p_threshold = min(p_min_local + 3.0, 1008.0)
        
        weight_sum = 0.0
        lat_weighted = 0.0
        lon_weighted = 0.0
        
        for ir in range(r_min, r_max):
            for ic in range(c_min, c_max):
                pv = mslp_det[ir, ic]
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
        wind_box = 5  # bán kính ~1.25 độ (~135 km), giữ RMW gần tâm - tránh bắt nhầm gió ngoại vi
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

        # Loại xoáy hoàn toàn lạnh (xoáy ngoại nhiệt đới, rãnh gió mùa lạnh) VÀ
        # siết ngưỡng tâm nóng: yêu cầu dị thường nóng RÕ RỆT >= +0.8°C ở >= 2
        # tầng (hoặc ở tầng duy nhất khả dụng). Ngưỡng cũ chỉ cần 1 tầng dương
        # thoáng qua nên dị thường nhiệt tầng cao rộng của môi trường cũng lọt.
        if warm_core_layers_checked > 0:
            need_positive = 2 if warm_core_layers_checked >= 2 else 1
            if warm_core_dt < 0.8 or warm_core_layers_positive < need_positive:
                is_warm_core = False

        # Đảm bảo khu vực trung tâm vẫn giữ được độ xoáy dương rõ rệt (không bị
        # phân rã) - yêu cầu > 1.0e-5 thay vì chỉ > 0 để loại xoáy yếu/nhiễu.
        if vort850 is not None:
            sub_vort_center = vort850[max(0, r0 - 3):min(ny, r0 + 4), max(0, c0 - 3):min(nx, c0 + 4)]
            if sub_vort_center.size > 0 and np.nanmax(sub_vort_center) <= 1.0e-5:
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

        # --- Gió tiếp tuyến đối xứng: bằng chứng QUAY THÀNH XOÁY thực sự ---
        vt_sym, vt_rmw_km = symmetric_tangential_wind(
            u10_grid, v10_grid, lats, lons, center_lat, center_lon)

        # Tâm sau tinh chỉnh có bị trôi lên địa hình cao không?
        center_on_terrain = _on_high_terrain(center_lat, center_lon)

        # Điều kiện tối thiểu để ghi nhận xoáy thuận / bão / ATNĐ
        # 1. Gió mạnh >= 10 m/s (~20 kt) BẮT BUỘC.
        # 2. GIÓ TIẾP TUYẾN ĐỐI XỨNG >= 6 m/s: hệ thống phải THỰC SỰ QUAY quanh
        #    tâm. Đây là điều kiện loại được ca sai trong ảnh - luồng gió mùa/gió
        #    địa hình chạy một chiều có gió max lớn nhưng gió tiếp tuyến ~ 0.
        # 3. Không phải xoáy ngoại nhiệt đới (cold core) và có xoáy dương 850hPa.
        # 4. Không phải vùng quá khô.
        # 5. Không nằm sâu trong lục địa châu Á VÀ không nằm trên địa hình núi cao
        #    (Đài Loan, Luzon, Nhật) - nơi áp mực biển bị ngoại suy sinh tâm giả.
        if (max_wind_ms >= 10.0 and vt_sym >= 6.0 and p_min_local <= 1008.0
                and is_warm_core and is_moist
                and not is_deep_inland(center_lat, center_lon)
                and not center_on_terrain):
            sys_info = classify_system(max_wind_kts, p_min_local)
            detected.append({
                "lat": round(float(center_lat), 2),
                "lon": round(float(center_lon), 2),
                "min_mslp_hpa": round(float(p_min_local), 1),
                "max_wind_kts": round(float(max_wind_kts), 1),
                "max_wind_ms": round(float(max_wind_ms), 1),
                "max_wind_kmh": round(float(max_wind_kmh), 1),
                "vt_sym_ms": round(float(vt_sym), 1),
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
        f"lev_surface=on&var_HGT=on&"  # độ cao địa hình (orography) để lọc tâm giả trên núi
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

        # 1b. Đọc độ cao địa hình (orography, mét) từ trường HGT ở mặt (surface) -
        # dùng làm mặt nạ loại tâm giả nằm trên núi cao. Nếu không có, để None và
        # thuật toán tự dùng fallback theo tọa độ (is_high_terrain).
        orog = None
        for _sn in ("orog", "gh", "hgt"):
            try:
                ds_o = xr.open_dataset(
                    grib_file, engine="cfgrib",
                    filter_by_keys={"typeOfLevel": "surface", "shortName": _sn},
                )
                _var = list(ds_o.data_vars)[0]
                orog = ds_o[_var].values
                ds_o.close()
                break
            except Exception:
                orog = None

        # 2. Đọc các tầng khí áp trên cao để tính toán tâm bão (chỉ xử lý trong RAM tại GitHub)
        u850, v850, t200, t300, t250, w500, rh700 = None, None, None, None, None, None, None
        try:
            # Dùng open_datasets (số nhiều) thay vì open_dataset: khi các biến U/V/T/W/RH
            # có số lượng level isobaricInhPa khác nhau, cfgrib không gộp được thành một
            # DataArray chung và toàn bộ dataset "chết ngầm" (rơi vào except bên dưới),
            # khiến mọi biến tầng cao đều thành None dù dữ liệu vẫn có trong file GRIB.
            datasets = xr.open_datasets(grib_file, engine="cfgrib")
            for ds_iso in datasets:
                if "isobaricInhPa" not in ds_iso.coords:
                    continue
                levels_available = (
                    list(ds_iso.isobaricInhPa.values)
                    if ds_iso.isobaricInhPa.ndim > 0
                    else [float(ds_iso.isobaricInhPa.values)]
                )
                # U, V ở 850 hPa (định vị xoáy)
                if "u" in ds_iso and 850 in levels_available:
                    u850 = ds_iso["u"].sel(isobaricInhPa=850).values
                    v850 = ds_iso["v"].sel(isobaricInhPa=850).values
                # Nhiệt độ T ở 300/250/200 hPa (kiểm tra Warm Core đa tầng)
                if "t" in ds_iso:
                    if 300 in levels_available:
                        t300 = ds_iso["t"].sel(isobaricInhPa=300).values - 273.15
                    if 250 in levels_available:
                        t250 = ds_iso["t"].sel(isobaricInhPa=250).values - 273.15
                    if 200 in levels_available:
                        t200 = ds_iso["t"].sel(isobaricInhPa=200).values - 273.15  # Kelvin -> Celsius
                # Vận tốc thẳng đứng w ở 500 hPa
                if "w" in ds_iso and 500 in levels_available:
                    w500 = ds_iso["w"].sel(isobaricInhPa=500).values
                # Độ ẩm tương đối RH ở 700 hPa
                if "r" in ds_iso and 700 in levels_available:
                    rh700 = ds_iso["r"].sel(isobaricInhPa=700).values
                ds_iso.close()
        except Exception as e:
            # Fallback nếu grib chia thành file con hoặc một số tầng chưa có
            print(f"Lưu ý: Không tải được tầng cao f{f_str} ({e}), dùng trường bề mặt + gradient áp suất để định vị.")

        # 3. TÍNH TOÁN VỊ TRÍ TÂM BÃO & THÔNG SỐ VẬT LÝ TẠI GITHUB
        cyclones = detect_cyclones_full(mslp, u10, v10, lats, lons, u850, v850, t200, t300, t250, w500, rh700, orog=orog)
        
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
