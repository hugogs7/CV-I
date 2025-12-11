"""
Visualization utilities for Lab 6: SIFT Feature Matching

This module contains helper functions for visualizing SIFT algorithm outputs.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch


def plot_gaussian_pyramid(gaussian_images, sigma=0.5, num_intervals=3):
    """Plot the actual generated Gaussian pyramid."""

    k = 2 ** (1 / num_intervals)
    num_octaves = len(gaussian_images)
    num_scales = len(gaussian_images[0])

    fig, axes = plt.subplots(num_octaves, num_scales, figsize=(2.5 * num_scales, 2.5 * num_octaves))

    for o in range(num_octaves):
        for i in range(num_scales):
            ax = axes[o, i]
            ax.imshow(gaussian_images[o][i], cmap='gray')
            ax.axis('off')

            relative_sigma = sigma * (k ** i)
            effective_sigma = relative_sigma * (2 ** o)
            ax.set_title(f'σ={relative_sigma:.1f}\neff={effective_sigma:.1f}', fontsize=8)

            if i == 0:
                h, w = gaussian_images[o][i].shape[:2]
                ax.set_ylabel(f'Oct {o}\n{w}×{h}', fontsize=10)

    plt.tight_layout()
    plt.show()


def visualize_all_dog_levels(dog_images):
    """Visualize DoG images for all octaves."""
    num_octaves = len(dog_images)
    num_dogs = len(dog_images[0])

    fig, axes = plt.subplots(num_octaves, num_dogs, figsize=(3 * num_dogs, 3 * num_octaves))

    for octave in range(num_octaves):
        for i in range(num_dogs):
            ax = axes[octave, i] if num_octaves > 1 else axes[i]

            dog = dog_images[octave][i]
            ax.imshow(dog, cmap='gray')
            ax.axis('off')
            ax.set_title(f'DoG[{i}]', fontsize=10)
    plt.suptitle('DoG pyramid: all octaves and scales', fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_keypoints(image, keypoints, sigma=0.5, num_intervals=3):
    """Visualize keypoints with size based on their scale."""

    k = 2 ** (1 / num_intervals)

    plt.figure(figsize=(12, 10))
    plt.imshow(image, cmap='gray')

    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for x, y, octave, scale_idx in keypoints:
        # Compute the effective sigma
        relative_sigma = sigma * (k ** scale_idx)
        effective_sigma = relative_sigma * (2 ** octave)

        radius = effective_sigma

        color = colors[octave % len(colors)]
        circle = plt.Circle((x, y), radius, fill=False, color=color, linewidth=1.5)
        plt.gca().add_patch(circle)

    plt.title(f'Keypoints: {len(keypoints)} detected')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_keypoints_with_orientation(image, keypoints, sigma=0.5, num_intervals=3):
    """Visualize keypoints with orientation arrows."""

    k = 2 ** (1 / num_intervals)

    plt.figure(figsize=(12, 10))
    plt.imshow(image, cmap='gray')

    colors = ['red', 'green', 'blue', 'orange', 'purple']

    for kp in keypoints:
        x, y, octave, scale_idx, angle = kp[:5]

        relative_sigma = sigma * (k ** scale_idx)
        effective_sigma = relative_sigma * (2 ** octave)
        radius = effective_sigma

        color = colors[octave % len(colors)]
        circle = plt.Circle((x, y), radius, fill=False, color=color, linewidth=1.5)
        plt.gca().add_patch(circle)

        # Draw orientation arrow
        arrow_length = radius * 1.5
        dx = arrow_length * np.cos(np.radians(angle))
        dy = arrow_length * np.sin(np.radians(angle))
        plt.arrow(x, y, dx, dy, color=color, width=0.5, head_width=3, head_length=3)

    plt.title(f'Keypoints with orientation: {len(keypoints)} detected')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def compare_matching_with_opencv(image, generateGaussianImages, generateGaussianKernels,
                                 dog_levels, findScaleSpaceExtremaWithDescriptors,
                                 sigma=0.5, num_intervals=3, rotate=True, rotation_angle=30):
    """Compare matching performance between your SIFT implementation and OpenCV."""

    k = 2 ** (1 / num_intervals)
    kernels = generateGaussianKernels(sigma, num_intervals)

    # Extract patch
    patch_x, patch_y = 100, 100
    patch_size = 150
    patch_original = image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size].copy()

    if rotate:
        h, w = patch_original.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        patch = cv2.warpAffine(patch_original, rotation_matrix, (w, h))
    else:
        patch = patch_original.copy()

    # Your SIFT implementation
    patch_gaussian_ours = generateGaussianImages(patch, kernels, num_octaves=3)
    patch_dog_ours = dog_levels(patch_gaussian_ours)
    patch_kp_ours = findScaleSpaceExtremaWithDescriptors(patch_dog_ours, patch_gaussian_ours,
                                                          sigma=sigma, num_intervals=num_intervals)

    orig_gaussian_ours = generateGaussianImages(patch_original, kernels, num_octaves=3)
    orig_dog_ours = dog_levels(orig_gaussian_ours)
    orig_kp_ours = findScaleSpaceExtremaWithDescriptors(orig_dog_ours, orig_gaussian_ours,
                                                         sigma=sigma, num_intervals=num_intervals)

    # Match descriptors
    matches_ours = []
    for kp1 in patch_kp_ours:
        x1, y1, o1, s1, a1, desc1 = kp1
        best_dist = float('inf')
        best_match = None

        for kp2 in orig_kp_ours:
            x2, y2, o2, s2, a2, desc2 = kp2
            dist = np.linalg.norm(desc1 - desc2)
            if dist < best_dist:
                best_dist = dist
                best_match = (x1, y1, x2, y2)

        if best_match and best_dist < 0.7:
            matches_ours.append(best_match)

    # OpenCV SIFT
    sift = cv2.SIFT_create()
    kp1_cv, desc1_cv = sift.detectAndCompute(patch, None)
    kp2_cv, desc2_cv = sift.detectAndCompute(patch_original, None)

    # Match OpenCV descriptors
    matches_cv = []
    if desc1_cv is not None and desc2_cv is not None and len(kp1_cv) >= 2 and len(kp2_cv) >= 2:
        bf = cv2.BFMatcher()
        matches_cv_raw = bf.knnMatch(desc1_cv, desc2_cv, k=2)

        for match in matches_cv_raw:
            if len(match) >= 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    pt1 = kp1_cv[m.queryIdx].pt
                    pt2 = kp2_cv[m.trainIdx].pt
                    matches_cv.append((pt1[0], pt1[1], pt2[0], pt2[1]))

    match_colors = ['red', 'cyan', 'yellow', 'magenta', 'lime', 'orange', 'white', 'pink']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top row: Your implementation
    axes[0, 0].imshow(patch, cmap='gray')
    for kp in patch_kp_ours:
        x, y, o, s, a, d = kp
        r = (sigma * (k ** s) * (2 ** o))
        circle = plt.Circle((x, y), r, fill=False, color='green', linewidth=1, alpha=0.5)
        axes[0, 0].add_patch(circle)
    axes[0, 0].set_title(f'Your SIFT: {"Rotated" if rotate else "Same"} Patch\n{len(patch_kp_ours)} keypoints')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(patch_original, cmap='gray')
    for kp in orig_kp_ours:
        x, y, o, s, a, d = kp
        r = (sigma * (k ** s) * (2 ** o))
        circle = plt.Circle((x, y), r, fill=False, color='blue', linewidth=1, alpha=0.5)
        axes[0, 1].add_patch(circle)
    axes[0, 1].set_title(f'Your SIFT: Original\n{len(orig_kp_ours)} keypoints\n{len(matches_ours)} matches')
    axes[0, 1].axis('off')

    for i, (x1, y1, x2, y2) in enumerate(matches_ours):
        color = match_colors[i % len(match_colors)]
        con = ConnectionPatch(xyA=(x1, y1), xyB=(x2, y2),
                              coordsA="data", coordsB="data",
                              axesA=axes[0, 0], axesB=axes[0, 1],
                              color=color, linewidth=3)
        fig.add_artist(con)
        axes[0, 0].scatter(x1, y1, s=10, c=color, marker='o', edgecolors='black', linewidths=2)
        axes[0, 1].scatter(x2, y2, s=10, c=color, marker='o', edgecolors='black', linewidths=2)

    # Bottom row: OpenCV
    axes[1, 0].imshow(patch, cmap='gray')
    for kp in kp1_cv:
        x, y = kp.pt
        r = kp.size / 2
        circle = plt.Circle((x, y), r, fill=False, color='green', linewidth=1, alpha=0.5)
        axes[1, 0].add_patch(circle)
    axes[1, 0].set_title(f'OpenCV SIFT: {"Rotated" if rotate else "Same"} Patch\n{len(kp1_cv)} keypoints')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(patch_original, cmap='gray')
    for kp in kp2_cv:
        x, y = kp.pt
        r = kp.size / 2
        circle = plt.Circle((x, y), r, fill=False, color='blue', linewidth=1, alpha=0.5)
        axes[1, 1].add_patch(circle)
    axes[1, 1].set_title(f'OpenCV SIFT: Original\n{len(kp2_cv)} keypoints\n{len(matches_cv)} matches')
    axes[1, 1].axis('off')

    for i, (x1, y1, x2, y2) in enumerate(matches_cv):
        color = match_colors[i % len(match_colors)]
        con = ConnectionPatch(xyA=(x1, y1), xyB=(x2, y2),
                              coordsA="data", coordsB="data",
                              axesA=axes[1, 0], axesB=axes[1, 1],
                              color=color, linewidth=3)
        fig.add_artist(con)
        axes[1, 0].scatter(x1, y1, s=10, c=color, marker='s', edgecolors='black', linewidths=2)
        axes[1, 1].scatter(x2, y2, s=10, c=color, marker='s', edgecolors='black', linewidths=2)

    title = f'Comparison: {"Rotated " + str(rotation_angle) + "°" if rotate else "No rotation"}'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def descriptor_exercise_with_gradients(image, computeGradients):
    """
    Visualize gradients and descriptors at multiple points for analysis.

    Demonstrates how gradients are computed and how they contribute to SIFT descriptors
    by showing gradient patterns and resulting histograms at different image locations.

    Args:
        image: Input grayscale image
        computeGradients: Function to compute image gradients
    """
    points = [
        (300, 300, 20, "Point A"),
        (200, 450, 20, "Point B"),
        (80, 80, 20, "Point C"),
        (300, 400, 20, "Point D"),
    ]

    magnitude, orientation = computeGradients(image)

    fig, axes = plt.subplots(len(points), 3, figsize=(15, 4*len(points)))

    for idx, (x, y, r, label) in enumerate(points):
        x, y, r = int(x), int(y), int(r)

        if y - r < 0 or y + r >= image.shape[0] or x - r < 0 or x + r >= image.shape[1]:
            continue

        mag_window = magnitude[y-r:y+r, x-r:x+r]
        ori_window = orientation[y-r:y+r, x-r:x+r]
        window = image[y-r:y+r, x-r:x+r]

        # Plot 1: Full image with highlighted region
        axes[idx, 0].imshow(image, cmap='gray')
        rect = plt.Rectangle((x-r, y-r), 2*r, 2*r, fill=False, edgecolor='green', linewidth=2)
        axes[idx, 0].add_patch(rect)
        circle = plt.Circle((x, y), r, fill=False, edgecolor='blue', linewidth=2)
        axes[idx, 0].add_patch(circle)
        axes[idx, 0].plot(x, y, 'r+', markersize=15, markeredgewidth=3)
        axes[idx, 0].set_title(f'{label}: ({x}, {y})')
        axes[idx, 0].axis('off')

        # Plot 2: Gradient visualization
        ax2 = axes[idx, 1]
        ax2.imshow(window, cmap='gray')

        h, w = window.shape
        cell_h, cell_w = h // 4, w // 4
        for i in range(1, 4):
            ax2.axhline(i * cell_h, color='green', linewidth=1, alpha=0.7)
            ax2.axvline(i * cell_w, color='green', linewidth=1, alpha=0.7)

        max_mag = mag_window.max() if mag_window.max() > 0 else 1

        for i in range(h):
            for j in range(w):
                mag = mag_window[i, j]
                ori = ori_window[i, j]

                length = (mag / max_mag)
                if length > 0.5:
                    dx = length * np.cos(np.radians(ori))
                    dy = length * np.sin(np.radians(ori))
                    ax2.arrow(j, i, dx, dy, head_width=1.5, head_length=1,
                             fc='red', ec='red', linewidth=1)

        ax2.set_title(f'Gradients\n(arrow = direction, length = strength)')
        ax2.set_xlim(-1, w+1)
        ax2.set_ylim(h+1, -1)
        ax2.axis('off')

        # Plot 3: Descriptor visualization
        ax3 = axes[idx, 2]

        num_cells = 4
        num_bins = 8
        cell_size = 2 * r // num_cells

        for ci in range(num_cells):
            for cj in range(num_cells):
                y_start = ci * cell_size
                y_end = (ci + 1) * cell_size
                x_start = cj * cell_size
                x_end = (cj + 1) * cell_size

                cell_mag = mag_window[y_start:y_end, x_start:x_end]
                cell_ori = ori_window[y_start:y_end, x_start:x_end]

                hist = np.zeros(num_bins)
                for oi in range(cell_mag.shape[0]):
                    for oj in range(cell_mag.shape[1]):
                        bin_idx = int(cell_ori[oi, oj] / 45) % num_bins
                        hist[bin_idx] += cell_mag[oi, oj]

                if hist.max() > 0:
                    hist = hist / hist.max()

                center_x = cj + 0.5
                center_y = ci + 0.5

                for b in range(num_bins):
                    if hist[b] > 0.1:
                        angle = b * 45
                        length = hist[b] * 0.4
                        dx = length * np.cos(np.radians(angle))
                        dy = length * np.sin(np.radians(angle))
                        ax3.arrow(center_x, center_y, dx, dy,
                                head_width=0.05, head_length=0.02,
                                fc='red', ec='red', linewidth=2, zorder=10)

        for i in range(5):
            ax3.axhline(i, color='green', linewidth=2)
            ax3.axvline(i, color='green', linewidth=2)

        ax3.set_xlim(-0.1, 4.1)
        ax3.set_ylim(4.1, -0.1)
        ax3.set_title(f'Descriptor\n(histogram of gradients per cell)')
        ax3.set_aspect('equal')
        ax3.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================================
# Transformation Functions for Robustness Testing
# ============================================================================

def apply_rotation(image, angle):
    """Rotate image by given angle in degrees."""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)


def apply_scale(image, scale_factor):
    """Scale image by given factor."""
    h, w = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    if scale_factor < 1.0:
        # Downscaling
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Pad to original size
        result = np.zeros((h, w), dtype=image.dtype)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = scaled
        return result
    else:
        # Upscaling
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # Crop to original size
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        return scaled[y_offset:y_offset+h, x_offset:x_offset+w]


def apply_brightness(image, brightness_change):
    """Add brightness change to image."""
    result = image.astype(np.float32) + brightness_change
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_noise(image, sigma):
    """Add Gaussian noise to image."""
    noise = np.random.normal(0, sigma, image.shape)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_blur(image, sigma):
    """Apply Gaussian blur to image."""
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_perspective(image, strength='slight'):
    """Apply perspective transformation."""
    h, w = image.shape

    if strength == 'slight':
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([[w*0.1, h*0.05], [w*0.95, h*0.1],
                             [w*0.9, h*0.95], [w*0.05, h*0.9]])
    else:  # strong
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([[w*0.2, h*0.15], [w*0.9, h*0.2],
                             [w*0.8, h*0.85], [w*0.1, h*0.75]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)


def compute_match_ratio(kp1, kp2, match_threshold=0.7):
    """Compute ratio of good matches between two keypoint sets."""
    if len(kp1) == 0 or len(kp2) == 0:
        return 0.0

    desc1 = np.array([kp[5] for kp in kp1])
    desc2 = np.array([kp[5] for kp in kp2])

    good_matches = 0
    for d1 in desc1:
        distances = np.linalg.norm(desc2 - d1, axis=1)
        sorted_idx = np.argsort(distances)
        if len(sorted_idx) > 1:
            if distances[sorted_idx[0]] < match_threshold * distances[sorted_idx[1]]:
                good_matches += 1

    return good_matches / min(len(kp1), len(kp2))


def visualize_transformations(image):
    """Visualize sample transformations applied to the image."""

    transformations = [
        ("Original", lambda img: img),
        ("Rotation 45°", lambda img: apply_rotation(img, 45)),
        ("Scale 0.5×", lambda img: apply_scale(img, 0.5)),
        ("Scale 2.0×", lambda img: apply_scale(img, 2.0)),
        ("Brightness +50", lambda img: apply_brightness(img, 50)),
        ("Darkness ×0.5", lambda img: (img * 0.5).astype(np.uint8)),
        ("Noise σ=25", lambda img: apply_noise(img, 25)),
        ("Blur σ=7", lambda img: apply_blur(img, 7)),
        ("Perspective", lambda img: apply_perspective(img, 'slight')),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (name, transform_func) in enumerate(transformations):
        transformed = transform_func(image.copy())
        axes[idx].imshow(transformed, cmap='gray')
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('Visual Comparison of Image Transformations',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def test_sift_robustness_with_opencv(image, generateGaussianImages, generateGaussianKernels,
                                      dog_levels, findScaleSpaceExtremaWithDescriptors,
                                      sigma=0.5, num_intervals=3, threshold=5):
    """
    Test SIFT robustness comparing student implementation vs OpenCV.

    Returns:
        tuple: (results_yours, results_opencv) - both are dicts with transformation names as keys
    """

    print("\nComputing keypoints for original image...")

    # YOUR IMPLEMENTATION
    kernels = generateGaussianKernels(sigma, num_intervals)
    gauss_orig = generateGaussianImages(image, kernels, num_octaves=4)
    dog_orig = dog_levels(gauss_orig)
    kp_yours_orig = findScaleSpaceExtremaWithDescriptors(dog_orig, gauss_orig, sigma, num_intervals, threshold)

    # OPENCV IMPLEMENTATION
    sift_opencv = cv2.SIFT_create()
    kp_opencv_orig, desc_opencv_orig = sift_opencv.detectAndCompute(image, None)
    # Convert to same format as yours
    kp_opencv_orig_formatted = [
        (kp.pt[0], kp.pt[1], 0, 0, kp.angle, desc_opencv_orig[i])
        for i, kp in enumerate(kp_opencv_orig)
    ]

    print(f"Original - Your SIFT: {len(kp_yours_orig)} keypoints")
    print(f"Original - OpenCV SIFT: {len(kp_opencv_orig_formatted)} keypoints\n")

    # Define transformations
    transformations = [
        ("Rotation 30°", lambda img: apply_rotation(img, 30)),
        ("Rotation 45°", lambda img: apply_rotation(img, 45)),
        ("Rotation 90°", lambda img: apply_rotation(img, 90)),
        ("Rotation 180°", lambda img: apply_rotation(img, 180)),
        ("Scale 0.5×", lambda img: apply_scale(img, 0.5)),
        ("Scale 1.5×", lambda img: apply_scale(img, 1.5)),
        ("Scale 2.0×", lambda img: apply_scale(img, 2.0)),
        ("Brightness +50", lambda img: apply_brightness(img, 50)),
        ("Darkness ×0.5", lambda img: (img * 0.5).astype(np.uint8)),
        ("Noise σ=10", lambda img: apply_noise(img, 10)),
        ("Noise σ=25", lambda img: apply_noise(img, 25)),
        ("Blur σ=3", lambda img: apply_blur(img, 3)),
        ("Blur σ=7", lambda img: apply_blur(img, 7)),
    ]

    results_yours = {}
    results_opencv = {}

    print("Testing transformations...")
    print("="*90)
    print(f"{'Transformation':<25} {'Your SIFT':<20} {'OpenCV SIFT':<20} {'Difference':<15}")
    print("="*90)

    for name, transform_func in transformations:
        # Apply transformation
        transformed = transform_func(image.copy())

        # YOUR SIFT
        gauss_trans = generateGaussianImages(transformed, kernels, num_octaves=4)
        dog_trans = dog_levels(gauss_trans)
        kp_yours_trans = findScaleSpaceExtremaWithDescriptors(dog_trans, gauss_trans, sigma, num_intervals, threshold)
        ratio_yours = compute_match_ratio(kp_yours_orig, kp_yours_trans)

        # OPENCV SIFT
        kp_opencv_trans, desc_opencv_trans = sift_opencv.detectAndCompute(transformed, None)
        if desc_opencv_trans is not None:
            kp_opencv_trans_formatted = [
                (kp.pt[0], kp.pt[1], 0, 0, kp.angle, desc_opencv_trans[i])
                for i, kp in enumerate(kp_opencv_trans)
            ]
            ratio_opencv = compute_match_ratio(kp_opencv_orig_formatted, kp_opencv_trans_formatted)
        else:
            ratio_opencv = 0.0

        results_yours[name] = ratio_yours
        results_opencv[name] = ratio_opencv

        # Print comparison
        diff = ratio_yours - ratio_opencv
        yours_status = "✓" if ratio_yours > 0.15 else "✗"
        opencv_status = "✓" if ratio_opencv > 0.15 else "✗"

        print(f"{name:<25} {ratio_yours:.3f} {yours_status:<15} {ratio_opencv:.3f} {opencv_status:<15} {diff:+.3f}")

    print("="*90)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ratios_yours = list(results_yours.values())
    ratios_opencv = list(results_opencv.values())
    names = list(results_yours.keys())

    colors_yours = ['green' if r > 0.15 else 'red' for r in ratios_yours]
    colors_opencv = ['green' if r > 0.15 else 'red' for r in ratios_opencv]

    y_pos = np.arange(len(names))

    # Plot 1: Your SIFT
    axes[0, 0].barh(y_pos, ratios_yours, color=colors_yours, alpha=0.7, edgecolor='black')
    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(names, fontsize=9)
    axes[0, 0].set_xlabel('Match Ratio', fontsize=11)
    axes[0, 0].set_title('Your SIFT Implementation', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=0.15, color='orange', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='x', alpha=0.3)
    axes[0, 0].set_xlim(0, 1.0)

    # Plot 2: OpenCV SIFT
    axes[0, 1].barh(y_pos, ratios_opencv, color=colors_opencv, alpha=0.7, edgecolor='black')
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(names, fontsize=9)
    axes[0, 1].set_xlabel('Match Ratio', fontsize=11)
    axes[0, 1].set_title('OpenCV SIFT (Reference)', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(x=0.15, color='orange', linestyle='--', linewidth=2, label='Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='x', alpha=0.3)
    axes[0, 1].set_xlim(0, 1.0)

    # Plot 3: Direct comparison
    x_pos = np.arange(len(names))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, ratios_yours, width, label='Your SIFT',
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x_pos + width/2, ratios_opencv, width, label='OpenCV SIFT',
                   color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Match Ratio', fontsize=11)
    axes[1, 0].set_title('Direct Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(y=0.15, color='orange', linestyle='--', linewidth=2)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim(0, 1.0)

    # Plot 4: Difference
    differences = [results_yours[k] - results_opencv[k] for k in names]
    colors_diff = ['green' if d > 0 else 'red' for d in differences]
    axes[1, 1].barh(y_pos, differences, color=colors_diff, alpha=0.7, edgecolor='black')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(names, fontsize=9)
    axes[1, 1].set_xlabel('Difference (Your - OpenCV)', fontsize=11)
    axes[1, 1].set_title('Performance Difference', fontsize=12, fontweight='bold')
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    avg_yours = np.mean(ratios_yours)
    avg_opencv = np.mean(ratios_opencv)

    pass_count_yours = sum(1 for r in ratios_yours if r > 0.15)
    pass_count_opencv = sum(1 for r in ratios_opencv if r > 0.15)

    print(f"Your SIFT:")
    print(f"  Average match ratio: {avg_yours:.3f}")
    print(f"  Tests passed: {pass_count_yours}/{len(transformations)} ({pass_count_yours/len(transformations)*100:.0f}%)")
    print(f"\nOpenCV SIFT:")
    print(f"  Average match ratio: {avg_opencv:.3f}")
    print(f"  Tests passed: {pass_count_opencv}/{len(transformations)} ({pass_count_opencv/len(transformations)*100:.0f}%)")
    print(f"\nDifference: {avg_yours - avg_opencv:+.3f}")
    print("="*70)

    return results_yours, results_opencv


def visualize_transformations_custom(image):
    """Visualize all 15 transformations applied to the image."""

    transformations = [
        ("Original", lambda img: img),
        ("Rotation 30°", lambda img: apply_rotation(img, 30)),
        ("Rotation 45°", lambda img: apply_rotation(img, 45)),
        ("Rotation 90°", lambda img: apply_rotation(img, 90)),
        ("Rotation 180°", lambda img: apply_rotation(img, 180)),
        ("Scale 0.5×", lambda img: apply_scale(img, 0.5)),
        ("Scale 1.5×", lambda img: apply_scale(img, 1.5)),
        ("Scale 2.0×", lambda img: apply_scale(img, 2.0)),
        ("Brightness +50", lambda img: apply_brightness(img, 50)),
        ("Darkness ×0.5", lambda img: (img * 0.5).astype(np.uint8)),
        ("Noise σ=10", lambda img: apply_noise(img, 10)),
        ("Noise σ=25", lambda img: apply_noise(img, 25)),
        ("Blur σ=3", lambda img: apply_blur(img, 3)),
        ("Blur σ=7", lambda img: apply_blur(img, 7)),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for idx, (name, transform_func) in enumerate(transformations):
        transformed = transform_func(image.copy())
        axes[idx].imshow(transformed, cmap='gray')
        axes[idx].set_title(name, fontsize=10, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle('Visual Comparison of All Image Transformations',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


def test_sift_robustness_with_opencv_custom(image, generateGaussianImages, generateGaussianKernels,
                                             dog_levels, findScaleSpaceExtremaWithDescriptors,
                                             sigma=0.5, num_intervals=3, threshold=5):
    """
    Test SIFT robustness comparing student implementation vs OpenCV.

    Returns:
        tuple: (results_yours, results_opencv) - both are dicts with transformation names as keys
    """

    print("\nComputing keypoints for original image...")

    # YOUR IMPLEMENTATION
    kernels = generateGaussianKernels(sigma, num_intervals)
    gauss_orig = generateGaussianImages(image, kernels, num_octaves=4)
    dog_orig = dog_levels(gauss_orig)
    kp_yours_orig = findScaleSpaceExtremaWithDescriptors(dog_orig, gauss_orig, sigma, num_intervals, threshold)

    # OPENCV IMPLEMENTATION
    sift_opencv = cv2.SIFT_create()
    kp_opencv_orig, desc_opencv_orig = sift_opencv.detectAndCompute(image, None)
    # Convert to same format as yours
    kp_opencv_orig_formatted = [
        (kp.pt[0], kp.pt[1], 0, 0, kp.angle, desc_opencv_orig[i])
        for i, kp in enumerate(kp_opencv_orig)
    ]

    print(f"Original - Your SIFT: {len(kp_yours_orig)} keypoints")
    print(f"Original - OpenCV SIFT: {len(kp_opencv_orig_formatted)} keypoints\n")

    # Define transformations
    transformations = [
        ("Rotation 30°", lambda img: apply_rotation(img, 30)),
        ("Rotation 45°", lambda img: apply_rotation(img, 45)),
        ("Rotation 90°", lambda img: apply_rotation(img, 90)),
        ("Rotation 180°", lambda img: apply_rotation(img, 180)),
        ("Scale 0.5×", lambda img: apply_scale(img, 0.5)),
        ("Scale 1.5×", lambda img: apply_scale(img, 1.5)),
        ("Scale 2.0×", lambda img: apply_scale(img, 2.0)),
        ("Brightness +50", lambda img: apply_brightness(img, 50)),
        ("Darkness ×0.5", lambda img: (img * 0.5).astype(np.uint8)),
        ("Noise σ=10", lambda img: apply_noise(img, 10)),
        ("Noise σ=25", lambda img: apply_noise(img, 25)),
        ("Blur σ=3", lambda img: apply_blur(img, 3)),
        ("Blur σ=7", lambda img: apply_blur(img, 7)),
    ]

    results_yours = {}
    results_opencv = {}

    print("Testing transformations...")
    print("="*90)
    print(f"{'Transformation':<25} {'Your SIFT':<20} {'OpenCV SIFT':<20} {'Difference':<15}")
    print("="*90)

    for name, transform_func in transformations:
        # Apply transformation
        transformed = transform_func(image.copy())

        # YOUR SIFT
        gauss_trans = generateGaussianImages(transformed, kernels, num_octaves=4)
        dog_trans = dog_levels(gauss_trans)
        kp_yours_trans = findScaleSpaceExtremaWithDescriptors(dog_trans, gauss_trans, sigma, num_intervals, threshold)
        ratio_yours = compute_match_ratio(kp_yours_orig, kp_yours_trans)

        # OPENCV SIFT
        kp_opencv_trans, desc_opencv_trans = sift_opencv.detectAndCompute(transformed, None)
        if desc_opencv_trans is not None:
            kp_opencv_trans_formatted = [
                (kp.pt[0], kp.pt[1], 0, 0, kp.angle, desc_opencv_trans[i])
                for i, kp in enumerate(kp_opencv_trans)
            ]
            ratio_opencv = compute_match_ratio(kp_opencv_orig_formatted, kp_opencv_trans_formatted)
        else:
            ratio_opencv = 0.0

        results_yours[name] = ratio_yours
        results_opencv[name] = ratio_opencv

        # Print comparison
        diff = ratio_yours - ratio_opencv
        yours_status = "✓" if ratio_yours > 0.15 else "✗"
        opencv_status = "✓" if ratio_opencv > 0.15 else "✗"

        print(f"{name:<25} {ratio_yours:.3f} {yours_status:<15} {ratio_opencv:.3f} {opencv_status:<15} {diff:+.3f}")

    print("="*90)

    # Summary statistics
    ratios_yours = list(results_yours.values())
    ratios_opencv = list(results_opencv.values())

    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    avg_yours = np.mean(ratios_yours)
    avg_opencv = np.mean(ratios_opencv)

    pass_count_yours = sum(1 for r in ratios_yours if r > 0.15)
    pass_count_opencv = sum(1 for r in ratios_opencv if r > 0.15)

    print(f"Your SIFT:")
    print(f"  Average match ratio: {avg_yours:.3f}")
    print(f"  Tests passed: {pass_count_yours}/{len(transformations)} ({pass_count_yours/len(transformations)*100:.0f}%)")
    print(f"\nOpenCV SIFT:")
    print(f"  Average match ratio: {avg_opencv:.3f}")
    print(f"  Tests passed: {pass_count_opencv}/{len(transformations)} ({pass_count_opencv/len(transformations)*100:.0f}%)")
    print(f"\nDifference: {avg_yours - avg_opencv:+.3f}")
    print("="*70)

    return results_yours, results_opencv