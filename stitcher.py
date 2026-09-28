"""Image Stitching 拼接引擎：階段二/三/四.

混合式流程 (Hybrid):
  1. SIFT + BFMatcher + Lowe ratio -> 相鄰影像相對單應性矩陣
  2. 以中間影像為基準 (reference)，鏈式累積全域 H
  3. 平移正規化 -> 大畫布 Warp
  4. 色彩平衡 ON : GainCompensator + MultiBandBlender
     色彩平衡 OFF: 簡單平均融合 (naive average, 無增益補償)
  5. 每張圖多邊形 / 縮放比例 / Yaw 近似角 (階段四)
  6. 手刻鏈失敗時 fallback 到 cv2.Stitcher_PANORAMA

對應 Project-Architecture.md 階段二、三、四。
"""
from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StitchResult:
    panorama: np.ndarray            # BGR uint8，已裁掉黑邊
    filenames: list[str] = field(default_factory=list)
    H_globals: list[np.ndarray] = field(default_factory=list)  # 含平移+裁切後對應 panorama 座標
    polygons: list[np.ndarray] = field(default_factory=list)   # 每張圖在 panorama 上的四角 (4,2)
    scales: list[float] = field(default_factory=list)
    yaws: list[float] = field(default_factory=list)            # 度，近似值
    gains: list[float] = field(default_factory=list)
    inliers: list[int] = field(default_factory=list)           # 每段相鄰匹配的 inlier 數
    method: str = "manual"          # manual | stitcher_fallback
    work_width: int = 1240
    canvas_size: tuple[int, int] = (0, 0)  # 裁切前 (W, H)


# ----------------------------------------------------------------------------
# 基礎工具
# ----------------------------------------------------------------------------
def load_images_sorted(folder: str, work_width: int = 1240) -> tuple[list[np.ndarray], list[str]]:
    patterns = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png", "*.BMP", "*.bmp")
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder, p)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"資料夾 {folder} 內找不到影像 (需 JPG/PNG/BMP)")
    images: list[np.ndarray] = []
    for f in files:
        img = cv2.imread(f)
        if img is None:
            raise IOError(f"無法讀取影像: {f}")
        h, w = img.shape[:2]
        if w > work_width:
            s = work_width / float(w)
            img = cv2.resize(img, (work_width, int(round(h * s))), interpolation=cv2.INTER_AREA)
        images.append(img)
    return images, [os.path.basename(f) for f in files]


def match_pair(gray_a: np.ndarray, gray_b: np.ndarray,
               sift, ratio: float = 0.75, ransac_thresh: float = 5.0,
               min_good: int = 10, mode: str = "homography") -> tuple[np.ndarray, int, int]:
    """估計 H_B_to_A：把 B 座標系的點投影到 A 座標系.

    mode="homography": findHomography + RANSAC (架構原文，8 自由度，含透視)；
    mode="affine": estimateAffinePartial2D + RANSAC 後補為 3x3
      （旋轉+等比縮放+平移，4 自由度；實測本組照片透視項會連乘發散，
       affine 鏈畫布 2650x1019、scale 0.9~1.1 才合理，故作為穩定後援）。
    Returns: (H, n_inliers, n_good)
    """
    ka, da = sift.detectAndCompute(gray_a, None)
    kb, db = sift.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        raise RuntimeError("特徵點不足 (SIFT 回傳空值)")
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(db, da, k=2)  # query=B, train=A
    good = [m for m, n in pairs if m.distance < ratio * n.distance]
    if len(good) < min_good:
        raise RuntimeError(f"良好匹配僅 {len(good)} (< {min_good})，重疊不足或紋理太少")
    # src = B 上的點, dst = A 上的點
    src = np.float32([kb[m.queryIdx].pt for m in good])
    dst = np.float32([ka[m.trainIdx].pt for m in good])
    if mode == "affine":
        A, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if A is None or mask is None:
            raise RuntimeError("estimateAffinePartial2D (RANSAC) 失敗")
        H = np.eye(3)
        H[:2, :] = A
    else:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        if H is None or mask is None:
            raise RuntimeError("findHomography (RANSAC) 失敗")
    return H, int(mask.sum()), len(good)


def homography_scale(H: np.ndarray) -> float:
    """由 H 線性部分的 |det| 開根號得縮放比例 (階段四)."""
    Hn = H / (H[2, 2] if abs(H[2, 2]) > 1e-9 else 1.0)
    M = Hn[:2, :2]
    return float(math.sqrt(abs(float(np.linalg.det(M)))))


def estimate_yaws(polygons: list[np.ndarray], ref_idx: int, work_width: int) -> list[float]:
    """Yaw 近似法 (已與使用者確認)：原地旋轉假設下，
    yaw_i = atan2(cx_i - cx_ref, f)，f 由水平視角約 50 度推估.

    f = (work_width / 2) / tan(25°). 中心圖 yaw=0，左右依水平位移展開.
    """
    f = (work_width / 2.0) / math.tan(math.radians(25.0))
    centers = [np.mean(p, axis=0) for p in polygons]
    cx_ref = float(centers[ref_idx][0])
    yaws = []
    for c in centers:
        yaws.append(float(math.degrees(math.atan2(float(c[0]) - cx_ref, f))))
    return yaws


# ----------------------------------------------------------------------------
# 手刻鏈
# ----------------------------------------------------------------------------
def _accumulate_to_reference(rel_H: list[np.ndarray], ref: int) -> list[np.ndarray]:
    """rel_H[i] : H_{i+1 -> i}. 回傳每張圖 H_{i -> ref}."""
    n = len(rel_H) + 1
    H_to_ref: list[np.ndarray] = [None] * n  # type: ignore
    H_to_ref[ref] = np.eye(3)
    for i in range(ref + 1, n):          # 右側：H_i->ref = H_ref->.. 累乘
        H_to_ref[i] = H_to_ref[i - 1] @ rel_H[i - 1]
    for i in range(ref - 1, -1, -1):     # 左側：H_i->ref = inv(rel_H[i]) @ H_{i+1->ref}
        H_to_ref[i] = np.linalg.inv(rel_H[i]) @ H_to_ref[i + 1]
    return H_to_ref


def _warp_and_masks(images: list[np.ndarray], H_globals: list[np.ndarray],
                    canvas_w: int, canvas_h: int):
    warped, masks = [], []
    for img, H in zip(images, H_globals):
        w = cv2.warpPerspective(img, H, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR, borderValue=0)
        h, ww = img.shape[:2]
        m = cv2.warpPerspective(np.full((h, ww), 255, np.uint8), H,
                                (canvas_w, canvas_h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
        warped.append(w)
        masks.append(m)
    return warped, masks


def _blend_gain_multiband(warped: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """Gain 補償 + MultiBandBlender，applies in place on copies."""
    h, w = warped[0].shape[:2]
    corners = [(0, 0)] * len(warped)
    sizes = [(w, h)] * len(warped)
    comp = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN)
    comp.feed(corners, warped, masks)
    gains = [float(np.asarray(comp.getMatGains()[i]).ravel()[0]) for i in range(len(warped))]
    compensated = [comp.apply(i, corners[i], warped[i], masks[i]) for i in range(len(warped))]
    blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_MULTI_BAND)
    try:  # num_bands 預設 5 在部分版本需 setNumBands
        blender.setNumBands(5)
    except Exception:
        pass
    blender.prepare(corners, sizes)
    for img, m in zip(compensated, masks):
        blender.feed(img.astype(np.int16), m, (0, 0))
    res, _ = blender.blend(None, None)
    pano = np.clip(res, 0, 255).astype(np.uint8)
    return pano, gains


def _blend_naive_average(warped: list[np.ndarray], masks: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    """色彩平衡 OFF：無增益補償，重疊區取算術平均."""
    acc = np.zeros_like(warped[0], dtype=np.float64)
    cnt = np.zeros(warped[0].shape[:2], dtype=np.float64)
    for img, m in zip(warped, masks):
        w8 = (m > 0).astype(np.float64)
        acc += img.astype(np.float64) * w8[..., None]
        cnt += w8
    cnt = np.maximum(cnt, 1.0)
    pano = (acc / cnt[..., None]).astype(np.uint8)
    return pano, [1.0] * len(warped)


def _crop_black_border(pano: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """以聯集 mask 的 boundingRect 裁掉外圍黑邊；回傳 (裁切後 pano, offset)."""
    union = np.zeros_like(masks[0])
    for m in masks:
        union = cv2.bitwise_or(union, m)
    ys, xs = np.where(union > 0)
    if len(xs) == 0:
        return pano, np.zeros(3)
    x, y, w, h = cv2.boundingRect(union)
    return pano[y:y + h, x:x + w], np.array([x, y])


def _chain_and_canvas(images: list[np.ndarray], rel_H: list[np.ndarray],
                      ref: int, max_pixels: int = 12000 * 8000
                      ) -> tuple[list[np.ndarray], list[np.ndarray], int, int, int, int]:
    """由相對 H 鏈式累積到 ref，並求畫布範圍；畫布不合理時 raise."""
    H_to_ref = _accumulate_to_reference(rel_H, ref)
    per_corners = []
    for idx, img in enumerate(images):
        h, w = img.shape[:2]
        c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        per_corners.append(cv2.perspectiveTransform(c, H_to_ref[idx]).reshape(-1, 2))
    all_pts = np.vstack(per_corners)
    xmin, ymin = np.floor(all_pts.min(0)).astype(int)
    xmax, ymax = np.ceil(all_pts.max(0)).astype(int)
    W, Hc = int(xmax - xmin), int(ymax - ymin)
    mean_area = float(np.mean([im.shape[0] * im.shape[1] for im in images]))
    if W <= 0 or Hc <= 0 or W * Hc > max_pixels or W * Hc > mean_area * len(images) * 8:
        raise RuntimeError(f"畫布尺寸異常 ({W}x{Hc})，可能某段 Homography 發散")
    return H_to_ref, per_corners, xmin, ymin, W, Hc


def stitch_manual(images: list[np.ndarray], filenames: list[str],
                  work_width: int, color_balance: bool,
                  ratio: float = 0.75) -> StitchResult:
    n = len(images)
    if n < 2:
        raise ValueError("至少需要 2 張影像")
    sift = cv2.SIFT_create()
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
    ref = n // 2

    last_err: Exception | None = None
    method_used = ""
    rel_H: list[np.ndarray] = []
    inlier_list: list[int] = []
    H_to_ref: list[np.ndarray] = []
    per_corners: list[np.ndarray] = []
    xmin = ymin = W = Hc = 0
    # 先照架構用 homography，發散時改用 affine 穩定鏈
    for mode in ("homography", "affine"):
        try:
            rel_H, inlier_list = [], []
            for i in range(n - 1):
                H, inl, _ngood = match_pair(grays[i], grays[i + 1], sift,
                                            ratio=ratio, mode=mode)
                rel_H.append(H)
                inlier_list.append(inl)
            H_to_ref, per_corners, xmin, ymin, W, Hc = _chain_and_canvas(images, rel_H, ref)
            method_used = f"manual_{mode}"
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if not method_used or not rel_H:
        raise RuntimeError(f"手刻鏈 (homography+affine) 皆失敗：{last_err}")

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=float)
    H_globals = [T @ H for H in H_to_ref]

    warped, masks = _warp_and_masks(images, H_globals, W, Hc)
    if color_balance:
        pano, gains = _blend_gain_multiband(warped, masks)
    else:
        pano, gains = _blend_naive_average(warped, masks)

    pano_cropped, offset = _crop_black_border(pano, masks)
    ox, oy = int(offset[0]), int(offset[1])

    polygons = [tc - np.array([xmin + ox, ymin + oy]) for tc in per_corners]
    # H_globals 對應到裁切後座標：左乘平移 (-ox,-oy)
    Tc = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], float)
    H_globals = [Tc @ H for H in H_globals]
    scales = [homography_scale(H) for H in H_to_ref]  # 縮放與平移無關，用未平移版
    yaws = estimate_yaws(polygons, ref, images[0].shape[1])

    return StitchResult(panorama=pano_cropped, filenames=filenames,
                        H_globals=H_globals, polygons=polygons,
                        scales=scales, yaws=yaws, gains=gains,
                        inliers=inlier_list, method=method_used,
                        work_width=work_width, canvas_size=(W, Hc))


# ----------------------------------------------------------------------------
# Fallback：cv2.Stitcher（拿不到逐張 H，多邊形以近似網格回填）
# ----------------------------------------------------------------------------
def stitch_fallback(images: list[np.ndarray], filenames: list[str],
                    work_width: int, color_balance: bool) -> StitchResult:
    mode = cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher_create(mode)
    # Stitcher 內部已有曝光補償；OFF 時嘗試關閉以體現差異
    if not color_balance:
        try:
            stitcher.setExposureCompensator(
                cv2.detail.ExposureCompensator_createDefault(
                    cv2.detail.ExposureCompensator_NO))
        except Exception:
            pass
    status, pano = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"cv2.Stitcher 失敗，狀態碼={status} (0=OK,1=需更多影像,2=Homography失敗,3=相機參數失敗)")
    n = len(images)
    H, W = pano.shape[:2]
    # 近似多邊形：沿寬度均分、20% 重疊，只能示意（method 標示 fallback）
    polygons, scales, yaws, Hg = [], [], [], []
    strip = W / (n * 0.8 + 0.2)
    for i in range(n):
        x0 = i * strip * 0.8
        poly = np.float32([[x0, 0], [x0 + strip, 0], [x0 + strip, H], [x0, H]])
        polygons.append(poly)
        scales.append(1.0)
        Hg.append(np.eye(3))
    f = (work_width / 2.0) / math.tan(math.radians(25.0))
    cx_ref = W / 2.0
    for poly in polygons:
        cx = float(poly[:, 0].mean())
        yaws.append(float(math.degrees(math.atan2(cx - cx_ref, f))))
    return StitchResult(panorama=pano, filenames=filenames, H_globals=Hg,
                        polygons=polygons, scales=scales, yaws=yaws,
                        gains=[1.0] * n, inliers=[], method="stitcher_fallback",
                        work_width=work_width, canvas_size=(W, H))


# ----------------------------------------------------------------------------
# 對外主入口
# ----------------------------------------------------------------------------
def stitch_folder(folder: str, work_width: int = 1240,
                  color_balance: bool = True, ratio: float = 0.75,
                  method: str = "auto") -> StitchResult:
    """method: auto=手刻優先、失敗轉 Stitcher；manual=只用手刻鏈；stitcher=只用 Stitcher."""
    images, filenames = load_images_sorted(folder, work_width)
    if method == "stitcher":
        return stitch_fallback(images, filenames, work_width, color_balance)
    try:
        return stitch_manual(images, filenames, work_width, color_balance, ratio)
    except Exception as e:
        if method == "manual":
            raise
        print(f"[stitcher] 手刻鏈失敗 ({e})，改用 cv2.Stitcher fallback ...")
        return stitch_fallback(images, filenames, work_width, color_balance)
