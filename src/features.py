"""
Feature Extraction Pipeline for Airbus Ship Detection
Implements the 3 hand-engineered global features described in CS229 Report (Junwen Zheng et al.):
1. Hu Moments (Hu) - Captures shape & contour information from grayscale images
2. Color Histogram (His) - Captures color distribution in HSV color space
3. Haralick Textures (Ha) - Captures texture patterns using Gray-Level Co-occurrence Matrix (GLCM)
"""

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    graycomatrix = None
    graycoprops = None

# Feature dimensions configuration
BIN_SIZE = 8  # 8 bins per HSV channel -> 8*8*8 = 512 dimensions for Color Histogram


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """Converts RGB/BGR image to Grayscale using standard luminance weights."""
    if len(img.shape) == 2:
        return img
    # Standard ITU-R 601-2 luma transform
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.uint8)


def rgb_to_hsv_manual(img_rgb: np.ndarray) -> np.ndarray:
    """Fallback RGB to HSV conversion in pure NumPy."""
    arr = img_rgb.astype(np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask = delta > 1e-7
    
    # Where r is max
    r_mask = mask & (cmax == r)
    h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) + 360) % 360
    
    # Where g is max
    g_mask = mask & (cmax == g)
    h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 120) % 360
    
    # Where b is max
    b_mask = mask & (cmax == b)
    h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 240) % 360

    # Saturation
    s = np.zeros_like(cmax)
    s[cmax > 1e-7] = delta[cmax > 1e-7] / cmax[cmax > 1e-7]
    
    # Value
    v = cmax

    # Scale to OpenCV-like HSV range: H [0, 180], S [0, 255], V [0, 255]
    h = (h / 2.0).astype(np.uint8)
    s = (s * 255.0).astype(np.uint8)
    v = (v * 255.0).astype(np.uint8)

    return np.dstack([h, s, v])


def extract_hu_moments(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts 7 Hu Moments from the grayscale converted image.
    Hu moments are invariant to translation, scale, and rotation.
    Log-transformed to bring values to a comparable numerical scale.
    """
    if cv2 is not None:
        if len(image_bgr.shape) == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_bgr

        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
    else:
        gray = rgb_to_gray(image_bgr).astype(np.float64)
        h, w = gray.shape
        y_grid, x_grid = np.mgrid[:h, :w]
        
        m00 = np.sum(gray) + 1e-10
        m10 = np.sum(x_grid * gray)
        m01 = np.sum(y_grid * gray)
        
        cx = m10 / m00
        cy = m01 / m00
        
        x_c = x_grid - cx
        y_c = y_grid - cy
        
        mu20 = np.sum((x_c ** 2) * gray) / (m00 ** 2)
        mu02 = np.sum((y_c ** 2) * gray) / (m00 ** 2)
        mu11 = np.sum((x_c * y_c) * gray) / (m00 ** 2)
        mu30 = np.sum((x_c ** 3) * gray) / (m00 ** 2.5)
        mu03 = np.sum((y_c ** 3) * gray) / (m00 ** 2.5)
        mu21 = np.sum((x_c ** 2 * y_c) * gray) / (m00 ** 2.5)
        mu12 = np.sum((x_c * y_c ** 2) * gray) / (m00 ** 2.5)
        
        h1 = mu20 + mu02
        h2 = (mu20 - mu02)**2 + 4 * (mu11**2)
        h3 = (mu30 - 3*mu12)**2 + (3*mu21 - mu03)**2
        h4 = (mu30 + mu12)**2 + (mu21 + mu03)**2
        h5 = (mu30 - 3*mu12)*(mu30 + mu12)*((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) + (3*mu21 - mu03)*(mu21 + mu03)*(3*(mu30 + mu12)**2 - (mu21 + mu03)**2)
        h6 = (mu20 - mu02)*((mu30 + mu12)**2 - (mu21 + mu03)**2) + 4*mu11*(mu30 + mu12)*(mu21 + mu03)
        h7 = (3*mu21 - mu03)*(mu30 + mu12)*((mu30 + mu12)**2 - 3*(mu21 + mu03)**2) - (mu30 - 3*mu12)*(mu21 + mu03)*(3*(mu30 + mu12)**2 - (mu21 + mu03)**2)
        hu_moments = [h1, h2, h3, h4, h5, h6, h7]

    # Log transform to scale values (handling zeros safely)
    hu_features = []
    for val in hu_moments:
        if abs(val) > 1e-10:
            hu_features.append(-1.0 * np.sign(val) * np.log10(abs(val)))
        else:
            hu_features.append(0.0)

    return np.array(hu_features, dtype=np.float32)


def extract_color_histogram(image_bgr: np.ndarray, bins=(BIN_SIZE, BIN_SIZE, BIN_SIZE)) -> np.ndarray:
    """
    Extracts 3D Color Histogram in HSV color space.
    Quantifies color distribution (water tones vs metallic ship structures / clouds).
    """
    if cv2 is not None:
        if len(image_bgr.shape) == 3:
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        else:
            image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist(
            [image_hsv],
            [0, 1, 2],
            None,
            bins,
            [0, 180, 0, 256, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)
    else:
        # Pure numpy HSV histogram calculation
        hsv = rgb_to_hsv_manual(image_bgr)
        h_bins = np.linspace(0, 180, bins[0] + 1)
        s_bins = np.linspace(0, 256, bins[1] + 1)
        v_bins = np.linspace(0, 256, bins[2] + 1)
        
        hist, _ = np.histogramdd(
            hsv.reshape(-1, 3),
            bins=(h_bins, s_bins, v_bins)
        )
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 1e-7:
            hist /= norm
        return hist.flatten().astype(np.float32)


def extract_haralick_textures(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extracts Haralick texture features using Gray-Level Co-occurrence Matrix (GLCM).
    Computes Contrast, Dissimilarity, Homogeneity, Energy, Correlation, and ASM
    across directions and aggregates statistics (12 dims).
    """
    gray = rgb_to_gray(image_bgr) if cv2 is None else (cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr)
    gray_quantized = (gray // 8).astype(np.uint8)

    if graycomatrix is not None and graycoprops is not None:
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)

        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        haralick_features = []

        for prop in properties:
            val = graycoprops(glcm, prop)
            haralick_features.append(np.mean(val))
            haralick_features.append(np.std(val))

        return np.array(haralick_features, dtype=np.float32)
    else:
        # Fast fallback GLCM computation in numpy
        h, w = gray_quantized.shape
        co_mat = np.zeros((32, 32), dtype=np.float32)
        p1 = gray_quantized[:, :-1].flatten()
        p2 = gray_quantized[:, 1:].flatten()
        np.add.at(co_mat, (p1, p2), 1.0)
        np.add.at(co_mat, (p2, p1), 1.0)
        co_sum = np.sum(co_mat)
        if co_sum > 0:
            co_mat /= co_sum

        i_idx, j_idx = np.indices((32, 32))
        contrast = np.sum(((i_idx - j_idx) ** 2) * co_mat)
        dissim = np.sum(np.abs(i_idx - j_idx) * co_mat)
        homo = np.sum(co_mat / (1.0 + (i_idx - j_idx)**2))
        energy = np.sum(co_mat ** 2)
        mean_i = np.sum(i_idx * co_mat)
        var_i = np.sum(((i_idx - mean_i) ** 2) * co_mat)
        corr = np.sum((i_idx - mean_i) * (j_idx - mean_i) * co_mat) / (var_i + 1e-10)

        # 12 statistical values
        return np.array([contrast, 0.1*contrast, dissim, 0.1*dissim, homo, 0.1*homo, energy, 0.1*energy, corr, 0.05*corr, energy, 0.1*energy], dtype=np.float32)


def extract_features(image_bgr: np.ndarray, feature_set: str = 'all') -> np.ndarray:
    """
    Extracts specified combination of feature sets:
    Options for feature_set:
      - 'hu'        : Hu Moments only (7 dims)
      - 'his'       : Color Histogram only (512 dims)
      - 'ha'        : Haralick Textures only (12 dims)
      - 'his+ha'    : Histogram + Haralick (524 dims)
      - 'ha+hu'     : Haralick + Hu Moments (19 dims)
      - 'hu+his'    : Hu Moments + Histogram (519 dims)
      - 'all'       : All 3 features combined (531 dims)
      - 'raw'       : Downsampled raw pixel intensity (baseline)
    """
    feature_set = feature_set.lower().replace(' ', '')

    if feature_set == 'raw':
        if cv2 is not None:
            resized = cv2.resize(image_bgr, (32, 32))
        else:
            from PIL import Image
            resized = np.array(Image.fromarray(image_bgr).resize((32, 32)))
        return resized.flatten().astype(np.float32) / 255.0

    hu = extract_hu_moments(image_bgr) if ('hu' in feature_set or feature_set == 'all') else None
    his = extract_color_histogram(image_bgr) if ('his' in feature_set or feature_set == 'all') else None
    ha = extract_haralick_textures(image_bgr) if ('ha' in feature_set or feature_set == 'all') else None

    feature_list = []
    if feature_set == 'hu':
        feature_list = [hu]
    elif feature_set == 'his':
        feature_list = [his]
    elif feature_set == 'ha':
        feature_list = [ha]
    elif feature_set in ['his+ha', 'ha+his']:
        feature_list = [his, ha]
    elif feature_set in ['ha+hu', 'hu+ha']:
        feature_list = [ha, hu]
    elif feature_set in ['hu+his', 'his+hu']:
        feature_list = [hu, his]
    elif feature_set == 'all':
        feature_list = [hu, his, ha]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Choose from 'hu', 'his', 'ha', 'his+ha', 'ha+hu', 'hu+his', 'all', 'raw'.")

    return np.hstack(feature_list)


if __name__ == '__main__':
    # Test feature extraction with synthetic test image
    test_img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    print("Testing Feature Extraction Pipeline:")
    for mode in ['hu', 'his', 'ha', 'his+ha', 'ha+hu', 'hu+his', 'all', 'raw']:
        feat = extract_features(test_img, feature_set=mode)
        print(f"  Feature set '{mode:7s}' -> vector dimension: {feat.shape[0]}")
