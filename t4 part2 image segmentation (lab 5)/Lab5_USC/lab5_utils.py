"""
Visualization utilities for Lab 5: Morphological Operations and Document Processing

This module contains helper functions for visualizing morphological operations,
OCR preprocessing pipelines, and document restoration results.
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def visualize_structuring_elements(kernels, names):
    """
    Display structuring elements (kernels) side by side.

    Args:
        kernels: List of numpy arrays representing structuring elements
        names: List of names for each kernel
    """
    n = len(kernels)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))

    if n == 1:
        axes = [axes]

    for ax, kernel, name in zip(axes, kernels, names):
        ax.imshow(kernel, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'{name}\n{kernel.shape}')

        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                ax.text(j, i, str(kernel[i, j]), ha='center', va='center',
                       color='red' if kernel[i, j] == 1 else 'blue', fontsize=12)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


def compare_morphological_ops(original, results_dict, titles=None):
    """
    Side-by-side comparison of morphological operation results.

    Args:
        original: Original binary image
        results_dict: Dictionary mapping operation names to result images
        titles: Optional custom titles (if None, uses dict keys)
    """
    n = len(results_dict) + 1
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    for idx, (name, img) in enumerate(results_dict.items(), start=1):
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(titles[idx-1] if titles else name)
        axes[idx].axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_morph_difference(original, processed, title='Difference'):
    """
    Show the difference between original and processed images.

    Args:
        original: Original binary image
        processed: Processed image after morphological operation
        title: Title for the difference plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title('Processed')
    axes[1].axis('off')

    diff = np.abs(original.astype(np.float32) - processed.astype(np.float32))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'{title}\n(pixels changed)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_ocr_pipeline(images_dict, ocr_results=None):
    """
    Show preprocessing steps with OCR results.

    Args:
        images_dict: Dictionary mapping step names to images
        ocr_results: Optional dictionary mapping step names to OCR text output
    """
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images_dict.items()):
        ax.imshow(img, cmap='gray')
        title = name
        if ocr_results and name in ocr_results:
            text = ocr_results[name][:50] + '...' if len(ocr_results.get(name, '')) > 50 else ocr_results.get(name, '')
            title += f'\nOCR: "{text}"'
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def compare_ocr_accuracy(results_dict, ground_truth=None):
    """
    Bar chart comparing OCR text extraction across different preprocessing methods.

    Args:
        results_dict: Dictionary mapping method names to extracted text
        ground_truth: Optional ground truth text for comparison
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(results_dict.keys())
    texts = [results_dict[m] for m in methods]

    if ground_truth:
        accuracies = []
        for text in texts:
            correct = sum(1 for a, b in zip(text.lower(), ground_truth.lower()) if a == b)
            acc = correct / max(len(ground_truth), 1) * 100
            accuracies.append(acc)

        bars = ax.bar(methods, accuracies, color='steelblue')
        ax.set_ylabel('Character Accuracy (%)')
        ax.set_title('OCR Accuracy by Preprocessing Method')
        ax.set_ylim(0, 100)

        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom')
    else:
        lengths = [len(text) for text in texts]
        bars = ax.bar(methods, lengths, color='steelblue')
        ax.set_ylabel('Characters Extracted')
        ax.set_title('OCR Output Length by Preprocessing Method')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Document Restoration Visualization Functions
# =============================================================================


def visualize_degradation_analysis(degraded, clean=None):
    """
    Analyze and visualize document degradation.

    Args:
        degraded: Degraded document image
        clean: Optional clean ground truth image
    """
    n = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    axes[0].imshow(degraded, cmap='gray')
    axes[0].set_title('Degraded Document')
    axes[0].axis('off')

    # Show histogram of degraded image
    axes[1].hist(degraded.ravel(), bins=256, range=(0, 256), color='steelblue', alpha=0.7)
    axes[1].set_title('Intensity Histogram')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')

    if clean is not None:
        axes[2].imshow(clean, cmap='gray')
        axes[2].set_title('Clean Ground Truth')
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Analyze noise
    if clean is not None:
        noise = np.abs(degraded.astype(np.float32) - clean.astype(np.float32))
        print(f"Noise statistics:")
        print(f"  Mean absolute error: {noise.mean():.2f}")
        print(f"  Max error: {noise.max():.2f}")
        print(f"  Affected pixels: {(noise > 0).sum()} ({100*(noise > 0).mean():.1f}%)")


def compare_restoration_steps(original, steps_dict, clean=None):
    """
    Compare document restoration at different steps.

    Args:
        original: Original degraded image
        steps_dict: Dictionary of {step_name: restored_image}
        clean: Optional clean ground truth for comparison
    """
    n = len(steps_dict) + 1 + (1 if clean is not None else 0)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    idx = 0
    axes[idx].imshow(original, cmap='gray')
    axes[idx].set_title('Original\n(Degraded)')
    axes[idx].axis('off')
    idx += 1

    for name, img in steps_dict.items():
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(name)
        axes[idx].axis('off')
        idx += 1

    if clean is not None:
        axes[idx].imshow(clean, cmap='gray')
        axes[idx].set_title('Ground Truth\n(Clean)')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_restoration_difference(degraded, restored, clean=None, title='Restoration'):
    """
    Show what was changed by the restoration process.

    Args:
        degraded: Original degraded image
        restored: Restored image
        clean: Optional clean ground truth
        title: Title for the plot
    """
    n = 4 if clean is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    axes[0].imshow(degraded, cmap='gray')
    axes[0].set_title('Degraded')
    axes[0].axis('off')

    axes[1].imshow(restored, cmap='gray')
    axes[1].set_title('Restored')
    axes[1].axis('off')

    diff = np.abs(degraded.astype(np.float32) - restored.astype(np.float32))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'{title}\nPixels Changed: {(diff > 0).sum()}')
    axes[2].axis('off')

    if clean is not None:
        error = np.abs(restored.astype(np.float32) - clean.astype(np.float32))
        axes[3].imshow(error, cmap='hot')
        axes[3].set_title(f'Remaining Error\nMSE: {(error**2).mean():.2f}')
        axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def plot_ocr_accuracy_comparison(degraded_text, restored_text, ground_truth):
    """
    Compare OCR accuracy before and after restoration.

    Args:
        degraded_text: Text extracted from degraded image
        restored_text: Text extracted from restored image
        ground_truth: Ground truth text
    """
    def char_accuracy(extracted, gt):
        if not gt:
            return 0
        matches = sum(1 for a, b in zip(extracted.lower(), gt.lower()) if a == b)
        return matches / len(gt) * 100

    def word_accuracy(extracted, gt):
        gt_words = gt.split()
        ext_words = extracted.split()
        if not gt_words:
            return 0
        matches = sum(1 for w in ext_words if w.lower() in [gw.lower() for gw in gt_words])
        return matches / len(gt_words) * 100

    degraded_char = char_accuracy(degraded_text, ground_truth)
    restored_char = char_accuracy(restored_text, ground_truth)
    degraded_word = word_accuracy(degraded_text, ground_truth)
    restored_word = word_accuracy(restored_text, ground_truth)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Character accuracy
    x = ['Degraded', 'Restored']
    char_vals = [degraded_char, restored_char]
    bars1 = axes[0].bar(x, char_vals, color=['#ff6b6b', '#51cf66'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Character-Level Accuracy')
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars1, char_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')

    # Word accuracy
    word_vals = [degraded_word, restored_word]
    bars2 = axes[1].bar(x, word_vals, color=['#ff6b6b', '#51cf66'])
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Word-Level Accuracy')
    axes[1].set_ylim(0, 105)
    for bar, val in zip(bars2, word_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')

    plt.suptitle(f'OCR Accuracy Improvement\nImprovement: +{restored_char - degraded_char:.1f}% (char), +{restored_word - degraded_word:.1f}% (word)')
    plt.tight_layout()
    plt.show()

    print(f"\nGround Truth: \"{ground_truth}\"")
    print(f"Degraded OCR: \"{degraded_text}\"")
    print(f"Restored OCR: \"{restored_text}\"")


def compute_psnr(restored, clean):
    """
    Compute Peak Signal-to-Noise Ratio between restored and clean images.

    Args:
        restored: Restored image
        clean: Clean ground truth image

    Returns:
        float: PSNR value in dB
    """
    mse = np.mean((restored.astype(np.float32) - clean.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(restored, clean, window_size=11, k1=0.01, k2=0.03):
    """
    Compute Structural Similarity Index between restored and clean images.

    Simplified implementation without gaussian weighting.

    Args:
        restored: Restored image
        clean: Clean ground truth image
        window_size: Size of the sliding window
        k1, k2: Constants for stability

    Returns:
        float: SSIM value in [0, 1]
    """
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2

    img1 = restored.astype(np.float64)
    img2 = clean.astype(np.float64)

    mu1 = cv2.blur(img1, (window_size, window_size))
    mu2 = cv2.blur(img2, (window_size, window_size))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.blur(img1 ** 2, (window_size, window_size)) - mu1_sq
    sigma2_sq = cv2.blur(img2 ** 2, (window_size, window_size)) - mu2_sq
    sigma12 = cv2.blur(img1 * img2, (window_size, window_size)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)


def plot_quality_metrics(degraded, restored, clean):
    """
    Visualize PSNR and SSIM improvements.

    Args:
        degraded: Original degraded image
        restored: Restored image
        clean: Clean ground truth image
    """
    psnr_degraded = compute_psnr(degraded, clean)
    psnr_restored = compute_psnr(restored, clean)
    ssim_degraded = compute_ssim(degraded, clean)
    ssim_restored = compute_ssim(restored, clean)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR comparison
    x = ['Degraded', 'Restored']
    psnr_vals = [psnr_degraded, psnr_restored]
    bars1 = axes[0].bar(x, psnr_vals, color=['#ff6b6b', '#51cf66'])
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Peak Signal-to-Noise Ratio\n(Higher is better)')
    for bar, val in zip(bars1, psnr_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f} dB', ha='center', va='bottom')

    # SSIM comparison
    ssim_vals = [ssim_degraded, ssim_restored]
    bars2 = axes[1].bar(x, ssim_vals, color=['#ff6b6b', '#51cf66'])
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('Structural Similarity Index\n(Higher is better, max=1.0)')
    axes[1].set_ylim(0, 1.1)
    for bar, val in zip(bars2, ssim_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', va='bottom')

    plt.suptitle(f'Image Quality Improvement\nPSNR: +{psnr_restored - psnr_degraded:.2f} dB, SSIM: +{ssim_restored - ssim_degraded:.4f}')
    plt.tight_layout()
    plt.show()

    return {
        'psnr_degraded': psnr_degraded,
        'psnr_restored': psnr_restored,
        'ssim_degraded': ssim_degraded,
        'ssim_restored': ssim_restored
    }


def plot_confusion_matrix(cm, class_names=['Background', 'Object']):
    """
    Display confusion matrix as a heatmap with annotations.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        class_names: Names for the classes
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted ' + class_names[0], 'Predicted ' + class_names[1]])
    ax.set_yticklabels(['Actual ' + class_names[0], 'Actual ' + class_names[1]])

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]:,}',
                   ha='center', va='center', color=text_color, fontsize=12)

    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()
    plt.show()


def visualize_segmentation_errors(image, prediction, ground_truth):
    """
    Color-coded visualization of TP, TN, FP, FN regions.

    Args:
        image: Original image (for overlay)
        prediction: Predicted binary mask
        ground_truth: Ground truth binary mask
    """
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = prediction & ground_truth
    tn = ~prediction & ~ground_truth
    fp = prediction & ~ground_truth
    fn = ~prediction & ground_truth

    h, w = prediction.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)

    result[tp] = [0, 255, 0]
    result[tn] = [128, 128, 128]
    result[fp] = [255, 0, 0]
    result[fn] = [0, 0, 255]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    axes[3].imshow(result)
    axes[3].set_title('Error Analysis\nGreen=TP, Gray=TN, Red=FP, Blue=FN')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    return tp.sum(), tn.sum(), fp.sum(), fn.sum()


def plot_metrics_comparison(metrics_dict):
    """
    Bar chart comparing different segmentation metrics.

    Args:
        metrics_dict: Dictionary mapping metric names to values
                     e.g., {'Accuracy': 0.95, 'Precision': 0.88, ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = list(metrics_dict.keys())
    values = [metrics_dict[m] for m in metrics]

    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    bars = ax.bar(metrics, values, color=colors)

    ax.set_ylabel('Score')
    ax.set_title('Segmentation Metrics Comparison')
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax.legend()

    plt.tight_layout()
    plt.show()


def visualize_ocr_comparison(original, processed_images, ocr_texts, ground_truth=None):
    """
    Comprehensive OCR comparison visualization.

    Args:
        original: Original image
        processed_images: Dict of {method_name: processed_image}
        ocr_texts: Dict of {method_name: extracted_text}
        ground_truth: Optional ground truth text
    """
    n = len(processed_images) + 1
    fig = plt.figure(figsize=(4 * min(n, 4), 6 * ((n + 3) // 4)))

    ax = fig.add_subplot((n + 3) // 4, min(n, 4), 1)
    ax.imshow(original, cmap='gray')
    ax.set_title('Original')
    ax.axis('off')

    for idx, (name, img) in enumerate(processed_images.items(), start=2):
        ax = fig.add_subplot((n + 3) // 4, min(n, 4), idx)
        ax.imshow(img, cmap='gray')
        text = ocr_texts.get(name, '')
        short_text = text[:30] + '...' if len(text) > 30 else text
        ax.set_title(f'{name}\n"{short_text}"', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    if ground_truth:
        print(f"\nGround Truth: \"{ground_truth}\"")
        print("-" * 50)
        for name, text in ocr_texts.items():
            print(f"{name}: \"{text}\"")


# =============================================================================
# Region Growing Visualization Functions
# =============================================================================

# =============================================================================
# Synthetic Degradation Functions for Educational Demos
# =============================================================================


def create_clean_text_image(text, font_size=1.5, image_size=(100, 400), thickness=2):
    """
    Create a clean binary text image for degradation experiments.

    Args:
        text: Text string to render
        font_size: OpenCV font scale
        image_size: Tuple (height, width) of output image
        thickness: Font thickness

    Returns:
        np.array: Binary image with white text on black background
    """
    img = np.zeros(image_size, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_size, thickness)[0]
    text_x = (image_size[1] - text_size[0]) // 2
    text_y = (image_size[0] + text_size[1]) // 2

    cv2.putText(img, text, (text_x, text_y), font, font_size, 255, thickness)

    return img


def add_salt_noise(image, ratio=0.01):
    """
    Add salt noise (random white pixels) to a binary image.

    Salt noise appears as white spots in black background areas.
    Can be removed with morphological OPENING.

    Args:
        image: Binary input image
        ratio: Fraction of pixels to corrupt (0.01 = 1%)

    Returns:
        np.array: Image with salt noise added
    """
    result = image.copy()
    num_salt = int(ratio * image.size)

    coords = [np.random.randint(0, i, num_salt) for i in image.shape]
    result[coords[0], coords[1]] = 255

    return result


def add_pepper_noise(image, ratio=0.01):
    """
    Add pepper noise (random black pixels) to a binary image.

    Pepper noise appears as black spots in white text areas.
    Can be removed with morphological CLOSING.

    Args:
        image: Binary input image
        ratio: Fraction of pixels to corrupt (0.01 = 1%)

    Returns:
        np.array: Image with pepper noise added
    """
    result = image.copy()
    num_pepper = int(ratio * image.size)

    coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
    result[coords[0], coords[1]] = 0

    return result


def apply_stroke_erosion(image, kernel_size=3, iterations=1):
    """
    Apply erosion to simulate stroke degradation (thinning/breaking).

    Erosion thins text strokes and can break thin connections.
    Can be partially repaired with morphological DILATION.

    Args:
        image: Binary input image
        kernel_size: Size of structuring element
        iterations: Number of erosion iterations

    Returns:
        np.array: Eroded image with thinner/broken strokes
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(image, kernel, iterations=iterations)


def create_degraded_text_demo(text="HELLO", show_plot=True):
    """
    Create a complete demonstration of different degradation types.

    This function creates clean text and applies various degradations,
    showing the educational relationship between degradation type and
    the morphological operation needed to fix it.

    Args:
        text: Text to render
        show_plot: Whether to display the visualization

    Returns:
        dict: Dictionary with 'clean', 'salt', 'pepper', 'eroded', 'combined' images
    """
    clean = create_clean_text_image(text)

    results = {
        'clean': clean,
        'salt_noise': add_salt_noise(clean, ratio=0.02),
        'pepper_noise': add_pepper_noise(clean, ratio=0.02),
        'eroded': apply_stroke_erosion(clean, kernel_size=3, iterations=1),
        'combined': add_salt_noise(add_pepper_noise(apply_stroke_erosion(clean, 2, 1), 0.01), 0.01)
    }

    if show_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        titles = [
            'Clean (Original)',
            'Salt Noise\n(Fix with OPENING)',
            'Pepper Noise\n(Fix with CLOSING)',
            'Eroded Strokes\n(Fix with DILATION)',
            'Combined Degradation\n(Fix with OPENING then CLOSING)',
            ''
        ]

        for idx, (name, img) in enumerate(results.items()):
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(titles[idx])
            axes[idx].axis('off')

        axes[5].axis('off')
        plt.suptitle('Synthetic Degradation Types and Their Morphological Fixes', fontsize=14)
        plt.tight_layout()
        plt.show()

    return results


# =============================================================================
# DIBCO Binarization Metrics
# =============================================================================


def compute_f_measure(prediction, ground_truth):
    """
    Compute F-measure (F1 score) for document binarization.

    F-measure is the harmonic mean of precision and recall.
    Standard metric used in DIBCO competitions.

    Args:
        prediction: Predicted binary image (0 = background, 255 = foreground/text)
        ground_truth: Ground truth binary image

    Returns:
        dict: {'precision': float, 'recall': float, 'f_measure': float}
    """
    pred = (prediction > 127).astype(bool)
    gt = (ground_truth > 127).astype(bool)

    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f_measure': f_measure
    }


def compute_pseudo_f_measure(prediction, ground_truth, skeleton_gt=None):
    """
    Compute pseudo F-measure (pFM) for document binarization.

    pFM uses skeleton-based evaluation to be less sensitive to
    stroke width variations. If no skeleton is provided, it uses
    a simple thinning of the ground truth.

    Args:
        prediction: Predicted binary image
        ground_truth: Ground truth binary image
        skeleton_gt: Optional pre-computed skeleton of ground truth

    Returns:
        float: Pseudo F-measure value
    """
    from skimage.morphology import skeletonize

    pred = (prediction > 127).astype(np.uint8)
    gt = (ground_truth > 127).astype(np.uint8)

    if skeleton_gt is None:
        skeleton_gt = skeletonize(gt > 0).astype(np.uint8)

    skeleton_pred = skeletonize(pred > 0).astype(np.uint8)

    tp_skeleton = np.sum(skeleton_pred & gt)
    skeleton_total = np.sum(skeleton_pred)
    gt_skeleton_total = np.sum(skeleton_gt)

    p_precision = tp_skeleton / skeleton_total if skeleton_total > 0 else 0
    p_recall = np.sum(skeleton_gt & pred) / gt_skeleton_total if gt_skeleton_total > 0 else 0

    pfm = 2 * p_precision * p_recall / (p_precision + p_recall) if (p_precision + p_recall) > 0 else 0

    return pfm


def compute_drd(prediction, ground_truth, block_size=8):
    """
    Compute Distance Reciprocal Distortion (DRD) metric.

    DRD measures the visual distortion for document images.
    It considers the distance of flipped pixels to nearest non-flipped pixels.
    Lower DRD is better (0 = perfect).

    Simplified implementation based on DIBCO evaluation.

    Args:
        prediction: Predicted binary image
        ground_truth: Ground truth binary image
        block_size: Block size for normalization

    Returns:
        float: DRD value (lower is better)
    """
    pred = (prediction > 127).astype(np.uint8)
    gt = (ground_truth > 127).astype(np.uint8)

    diff = np.abs(pred.astype(int) - gt.astype(int))

    nubn = 0
    h, w = gt.shape
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gt[i:i+block_size, j:j+block_size]
            if np.any(block == 1) and np.any(block == 0):
                nubn += 1

    if nubn == 0:
        return 0.0

    drd = np.sum(diff) / nubn

    return drd


def compute_all_binarization_metrics(prediction, ground_truth):
    """
    Compute all standard DIBCO binarization metrics.

    Args:
        prediction: Predicted binary image
        ground_truth: Ground truth binary image

    Returns:
        dict: Dictionary with all metrics
    """
    fm_metrics = compute_f_measure(prediction, ground_truth)
    psnr = compute_psnr(prediction, ground_truth)
    drd = compute_drd(prediction, ground_truth)

    return {
        'F-measure': fm_metrics['f_measure'],
        'Precision': fm_metrics['precision'],
        'Recall': fm_metrics['recall'],
        'PSNR': psnr,
        'DRD': drd
    }


def visualize_binarization_comparison(original, binarized, ground_truth, method_name=''):
    """
    Comprehensive visualization for binarization evaluation.

    Args:
        original: Original grayscale document image
        binarized: Binarized result
        ground_truth: Ground truth binary image
        method_name: Name of the binarization method
    """
    metrics = compute_all_binarization_metrics(binarized, ground_truth)

    pred = (binarized > 127).astype(bool)
    gt = (ground_truth > 127).astype(bool)

    error_map = np.zeros((*pred.shape, 3), dtype=np.uint8)
    error_map[pred & gt] = [0, 255, 0]
    error_map[~pred & ~gt] = [200, 200, 200]
    error_map[pred & ~gt] = [255, 0, 0]
    error_map[~pred & gt] = [0, 0, 255]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Document')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(binarized, cmap='gray')
    axes[0, 1].set_title(f'Binarized ({method_name})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(ground_truth, cmap='gray')
    axes[0, 2].set_title('Ground Truth')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(error_map)
    axes[1, 0].set_title('Error Map\nGreen=TP, Gray=TN, Red=FP, Blue=FN')
    axes[1, 0].axis('off')

    diff = np.abs(original.astype(float) - binarized.astype(float))
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Pixel Changes')
    axes[1, 1].axis('off')

    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['#2ecc71' if n != 'DRD' else '#e74c3c' for n in metric_names]

    bars = axes[1, 2].bar(metric_names, metric_values, color=colors)
    axes[1, 2].set_title('Binarization Metrics')
    axes[1, 2].set_ylabel('Value')

    for bar, val, name in zip(bars, metric_values, metric_names):
        fmt = '.4f' if name in ['F-measure', 'Precision', 'Recall'] else '.2f'
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:{fmt}}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Binarization Evaluation: {method_name}', fontsize=14)
    plt.tight_layout()
    plt.show()

    return metrics


def compare_binarization_methods(original, methods_dict, ground_truth):
    """
    Compare multiple binarization methods side by side.

    Args:
        original: Original grayscale document
        methods_dict: Dict of {method_name: binarized_image}
        ground_truth: Ground truth binary image

    Returns:
        dict: Metrics for each method
    """
    all_metrics = {}

    n = len(methods_dict) + 2
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    for idx, (name, img) in enumerate(methods_dict.items(), start=1):
        axes[idx].imshow(img, cmap='gray')
        metrics = compute_f_measure(img, ground_truth)
        axes[idx].set_title(f'{name}\nF={metrics["f_measure"]:.3f}')
        axes[idx].axis('off')
        all_metrics[name] = compute_all_binarization_metrics(img, ground_truth)

    axes[len(methods_dict) + 1].imshow(ground_truth, cmap='gray')
    axes[len(methods_dict) + 1].set_title('Ground Truth')
    axes[len(methods_dict) + 1].axis('off')

    for idx in range(len(methods_dict) + 2, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()

    return all_metrics


# =============================================================================
# Region Growing Visualization Functions
# =============================================================================

from collections import deque


def get_neighbors(image, p):
    """
    Get 4-connected neighbor coordinates for a pixel.

    Args:
        image: Image array (used for bounds checking)
        p: Tuple (row, col) of the pixel position

    Returns:
        list: List of (row, col) tuples for valid neighbors (up, down, left, right)
    """
    neighbors = []
    for delta in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
        neighbor = (p[0] + delta[0], p[1] + delta[1])
        if 0 <= neighbor[0] < image.shape[0] and 0 <= neighbor[1] < image.shape[1]:
            neighbors.append(neighbor)
    return neighbors


def region_growing_interactive(image, seed_points, similarity_criterion):
    """
    Region growing segmentation generator for iterative visualization.

    This function uses yield to return intermediate states, allowing
    visualization of the algorithm's progression step by step.

    Args:
        image: Grayscale image
        seed_points: Tuple of (row, col) seed point coordinates
                    e.g., ((100, 100), (200, 200))
        similarity_criterion: Function(pixel_value, region_id, labels) -> bool
                             Returns True if pixel should join the region

    Yields:
        np.array: Current state of the labels array
                 -1 = unassigned, 0+ = region ID
    """
    result = -1 * np.ones(image.shape, dtype=int)
    region_id = 0
    queue = deque()

    for s in seed_points:
        result[s] = region_id
        queue.append((s, region_id))
        region_id += 1

    while queue:
        p, rid = queue.popleft()
        for n in get_neighbors(image, p):
            if result[n] == -1 and similarity_criterion(image[n], rid, result):
                result[n] = rid
                queue.append((n, rid))
        yield result


def visualize_region_growing(image, seed_points, tau=15, interval=2000, cmap='nipy_spectral'):
    """
    Run and visualize region growing interactively.

    This is a convenience wrapper that handles the visualization loop.

    Args:
        image: Grayscale image to segment
        seed_points: Tuple of (row, col) seed point coordinates
        tau: Intensity threshold for similarity criterion
        interval: Show visualization every N iterations
        cmap: Colormap for the segmentation result

    Returns:
        np.array: Final segmentation result (labels array)
    """
    seed_intensities = {}
    for i, seed in enumerate(seed_points):
        seed_intensities[i] = image[seed]
        print(f"Seed {i} at {seed}: intensity = {image[seed]}")

    def similarity_criterion(pixel_value, region_id, labels):
        if region_id not in seed_intensities:
            return False
        return abs(int(pixel_value) - int(seed_intensities[region_id])) < tau

    print(f"\nRunning region growing (tau={tau}, visualizing every {interval} iterations)...")

    result = None
    for i, result in enumerate(region_growing_interactive(image, seed_points, similarity_criterion)):
        if i % interval == 0 and i > 0:
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.title('Original Image')
            for j, s in enumerate(seed_points):
                plt.plot(s[1], s[0], 'ro', markersize=10)
                plt.text(s[1]+5, s[0]-5, f'S{j}', color='red', fontsize=12)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(result, cmap=cmap, vmin=-1, vmax=len(seed_points))
            plt.title(f'Region Growing - Iteration {i}')
            plt.colorbar(label='Region ID (-1 = unassigned)')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    # Show final result
    if result is not None:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        for j, s in enumerate(seed_points):
            plt.plot(s[1], s[0], 'ro', markersize=10)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(result, cmap=cmap, vmin=-1, vmax=len(seed_points))
        plt.title(f'Final Segmentation - Iteration {i}')
        plt.colorbar(label='Region ID')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print(f"\nTotal iterations: {i}")
        print(f"Unique region labels: {np.unique(result)}")

    return result


# =============================================================================
# SVHN Dataset Loading Functions
# =============================================================================


def load_svhn_cropped(mat_path='data/test_32x32.mat', n_samples=10, seed=42):
    """
    Load cropped 32x32 digit images from SVHN dataset.

    The SVHN (Street View House Numbers) dataset contains real-world images
    of house numbers captured from Google Street View. The cropped version
    contains individual digits centered in 32x32 pixel images.

    Args:
        mat_path: Path to test_32x32.mat file
        n_samples: Number of random samples to load
        seed: Random seed for reproducibility

    Returns:
        tuple: (images, labels) where:
            - images: List of grayscale numpy arrays (32x32)
            - labels: List of digit labels (0-9)
    """
    import scipy.io

    data = scipy.io.loadmat(mat_path)
    X = data['X']  # Shape: (32, 32, 3, N) - RGB images
    y = data['y'].flatten()  # Labels: 1-10 where 10 represents 0

    # Fix labels: SVHN uses 10 for digit 0
    y = np.where(y == 10, 0, y)

    # Select random samples
    np.random.seed(seed)
    n_total = X.shape[3]
    indices = np.random.choice(n_total, min(n_samples, n_total), replace=False)

    images = []
    labels = []
    for idx in indices:
        # Extract image and convert RGB to grayscale
        img_rgb = X[:, :, :, idx]
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        images.append(img_gray)
        labels.append(int(y[idx]))

    return images, labels


def prepare_svhn_for_morphology(image, target_size=(100, 100), invert=True):
    """
    Prepare an SVHN image for morphological operations demonstration.

    Resizes the image for better visualization and binarizes it using
    Otsu's method. Optionally inverts so digit is white on black background.

    Args:
        image: 32x32 grayscale SVHN image
        target_size: Output size tuple (height, width) for visualization
        invert: If True, produces white digit on black background

    Returns:
        np.array: Binary image ready for morphological operations
    """
    # Resize for better visualization
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)

    # Binarize using Otsu's method
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(resized, 0, 255, thresh_type + cv2.THRESH_OTSU)

    return binary


def download_svhn_full(dest_dir='data/svhn_full'):
    """
    Download and extract the full SVHN test dataset.

    The full dataset contains original images with multiple house numbers
    in their natural context - more challenging than cropped version.

    Args:
        dest_dir: Destination directory for downloaded files

    Returns:
        str: Path to the extracted test images directory
    """
    import urllib.request
    import tarfile
    import os

    url = 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'
    tar_path = os.path.join(dest_dir, 'test.tar.gz')
    extracted_dir = os.path.join(dest_dir, 'test')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    if not os.path.exists(extracted_dir):
        print("Downloading SVHN full dataset (~276 MB)...")
        print(f"URL: {url}")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete. Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(dest_dir)
        os.remove(tar_path)
        print(f"Done! Images saved to: {extracted_dir}")
    else:
        print(f"SVHN full dataset already exists at: {extracted_dir}")

    return extracted_dir


def load_svhn_full_samples(svhn_dir='data/svhn_full/test', n_samples=5):
    """
    Load sample images from the full SVHN dataset.

    These are full-resolution images containing house numbers in context,
    with complex backgrounds, varying lighting, and multiple digits.

    Args:
        svhn_dir: Path to extracted SVHN test directory
        n_samples: Number of images to load

    Returns:
        list: List of BGR images (numpy arrays)
    """
    import os
    import glob

    pattern = os.path.join(svhn_dir, '*.png')
    image_files = sorted(glob.glob(pattern))

    if not image_files:
        print(f"No images found in {svhn_dir}")
        print("Run download_svhn_full() first to download the dataset.")
        return []

    # Load requested number of samples
    images = []
    for img_path in image_files[:n_samples]:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)

    print(f"Loaded {len(images)} images from SVHN full dataset")
    return images


def visualize_svhn_samples(images, labels=None, title='SVHN Samples'):
    """
    Display a grid of SVHN digit images.

    Args:
        images: List of grayscale or color images
        labels: Optional list of digit labels
        title: Title for the figure
    """
    n = len(images)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, img in enumerate(images):
        if len(img.shape) == 3:
            # Color image - convert BGR to RGB for display
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap='gray')

        if labels is not None and i < len(labels):
            axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SVHN Morphological Pipeline Evaluation
# =============================================================================


def evaluate_morphological_pipeline(pipeline_func, svhn_path='data/test_32x32.mat',
                                     n_samples=20, seed=42, degradation_ratio=0.02,
                                     verbose=True, show_examples=True, show_only_changed=True):
    """
    Evaluate a student's morphological preprocessing pipeline on SVHN digits.

    This function loads SVHN digits, applies synthetic degradation (salt, pepper,
    erosion, or combined), then runs the student's pipeline to restore them.
    OCR is used to measure accuracy before and after preprocessing.

    Args:
        pipeline_func: A function with signature:
                      pipeline_func(binary_image, degradation_type) -> restored_image
                      where degradation_type is one of: 'salt', 'pepper', 'eroded', 'combined'
        svhn_path: Path to SVHN test_32x32.mat file
        n_samples: Number of SVHN digits to test (more = slower but more reliable)
        seed: Random seed for reproducibility
        degradation_ratio: Intensity of synthetic degradation (0.01-0.05 typical)
        verbose: Print detailed results
        show_examples: Display visual examples of restoration
        show_only_changed: If True, only show examples where morphology changed the OCR result

    Returns:
        dict: Evaluation results with keys:
            - 'accuracy_before': Dict of accuracy by degradation type (before pipeline)
            - 'accuracy_after': Dict of accuracy by degradation type (after pipeline)
            - 'improvement': Dict of accuracy improvement by degradation type
            - 'overall_before': Overall accuracy before pipeline
            - 'overall_after': Overall accuracy after pipeline
            - 'overall_improvement': Overall accuracy improvement
            - 'details': List of per-sample results
    """
    # Try to import easyocr
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    except ImportError:
        print("EasyOCR not installed. Run: pip install easyocr")
        return None

    # Load SVHN data
    images, labels = load_svhn_cropped(svhn_path, n_samples=n_samples, seed=seed)

    if verbose:
        print(f"Loaded {len(images)} SVHN digits for evaluation")
        print(f"Testing degradation types: salt, pepper, eroded, combined")
        print("-" * 60)

    # Prepare binary versions
    binary_digits = [prepare_svhn_for_morphology(img, target_size=(100, 100)) for img in images]

    # Define degradation functions
    def apply_degradation(img, dtype, ratio):
        if dtype == 'salt':
            return add_salt_noise(img, ratio)
        elif dtype == 'pepper':
            return add_pepper_noise(img, ratio)
        elif dtype == 'eroded':
            return apply_stroke_erosion(img, kernel_size=2, iterations=1)
        elif dtype == 'combined':
            eroded = apply_stroke_erosion(img, kernel_size=2, iterations=1)
            with_pepper = add_pepper_noise(eroded, ratio * 0.5)
            return add_salt_noise(with_pepper, ratio * 0.5)
        return img

    def extract_digit(image):
        """Extract single digit from image using OCR."""
        results = reader.readtext(image)
        text = ''.join([r[1] for r in results]).strip()
        # Extract first digit found
        for char in text:
            if char.isdigit():
                return char
        return ''

    degradation_types = ['salt', 'pepper', 'eroded', 'combined']
    results = {
        'accuracy_before': {d: 0 for d in degradation_types},
        'accuracy_after': {d: 0 for d in degradation_types},
        'counts': {d: 0 for d in degradation_types},
        'details': []
    }

    examples = {d: {'original': None, 'degraded': None, 'restored': None,
                    'label': None, 'before': None, 'after': None} for d in degradation_types}

    for i, (binary, label) in enumerate(zip(binary_digits, labels)):
        ground_truth = str(label)

        for dtype in degradation_types:
            # Apply degradation
            degraded = apply_degradation(binary, dtype, degradation_ratio)

            # Apply student's pipeline
            try:
                restored = pipeline_func(degraded, dtype)
                if restored is None:
                    restored = degraded  # Fallback if pipeline returns None
            except Exception as e:
                if verbose:
                    print(f"Pipeline error on sample {i}, {dtype}: {e}")
                restored = degraded

            # OCR before and after
            ocr_before = extract_digit(degraded)
            ocr_after = extract_digit(restored)

            # Check accuracy
            correct_before = (ocr_before == ground_truth)
            correct_after = (ocr_after == ground_truth)

            results['counts'][dtype] += 1
            if correct_before:
                results['accuracy_before'][dtype] += 1
            if correct_after:
                results['accuracy_after'][dtype] += 1

            # Save example for visualization
            # If show_only_changed, only save examples where OCR result changed
            ocr_changed = (ocr_before != ocr_after)
            should_save = (examples[dtype]['original'] is None and
                          (not show_only_changed or ocr_changed))
            if should_save:
                examples[dtype] = {
                    'original': binary,
                    'degraded': degraded,
                    'restored': restored,
                    'label': ground_truth,
                    'before': ocr_before,
                    'after': ocr_after,
                    'changed': ocr_changed
                }

            results['details'].append({
                'sample': i,
                'degradation': dtype,
                'ground_truth': ground_truth,
                'ocr_before': ocr_before,
                'ocr_after': ocr_after,
                'correct_before': correct_before,
                'correct_after': correct_after
            })

    # Calculate percentages
    for dtype in degradation_types:
        count = results['counts'][dtype]
        results['accuracy_before'][dtype] = results['accuracy_before'][dtype] / count * 100
        results['accuracy_after'][dtype] = results['accuracy_after'][dtype] / count * 100

    # Calculate improvements
    results['improvement'] = {
        d: results['accuracy_after'][d] - results['accuracy_before'][d]
        for d in degradation_types
    }

    # Overall metrics
    total = sum(results['counts'].values())
    total_correct_before = sum(1 for d in results['details'] if d['correct_before'])
    total_correct_after = sum(1 for d in results['details'] if d['correct_after'])

    results['overall_before'] = total_correct_before / total * 100
    results['overall_after'] = total_correct_after / total * 100
    results['overall_improvement'] = results['overall_after'] - results['overall_before']

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print("MORPHOLOGICAL PIPELINE EVALUATION RESULTS")
        print("=" * 60)
        print(f"\n{'Degradation':<12} | {'Before':<10} | {'After':<10} | {'Change':<10}")
        print("-" * 50)
        for dtype in degradation_types:
            before = results['accuracy_before'][dtype]
            after = results['accuracy_after'][dtype]
            change = results['improvement'][dtype]
            sign = '+' if change >= 0 else ''
            print(f"{dtype:<12} | {before:>8.1f}% | {after:>8.1f}% | {sign}{change:>7.1f}%")

        print("-" * 50)
        print(f"{'OVERALL':<12} | {results['overall_before']:>8.1f}% | "
              f"{results['overall_after']:>8.1f}% | "
              f"{'+' if results['overall_improvement'] >= 0 else ''}{results['overall_improvement']:>7.1f}%")
        print("=" * 60)

        if results['overall_improvement'] > 5:
            print("\nPipeline shows significant improvement!")
        elif results['overall_improvement'] > 0:
            print("\nPipeline shows modest improvement.")
        elif results['overall_improvement'] == 0:
            print("\nPipeline shows no change in accuracy.")
        else:
            print("\nWarning: Pipeline decreased accuracy. Check your implementation.")

    # Show visual examples
    if show_examples:
        fig, axes = plt.subplots(4, 4, figsize=(16, 14))

        for row, dtype in enumerate(degradation_types):
            ex = examples[dtype]

            # Handle case where no changed example was found
            if ex['original'] is None:
                for col in range(4):
                    axes[row, col].text(0.5, 0.5, f'No example with\nchanged OCR result\nfor {dtype.upper()}',
                                       ha='center', va='center', fontsize=12,
                                       transform=axes[row, col].transAxes)
                    axes[row, col].axis('off')
                continue

            axes[row, 0].imshow(ex['original'], cmap='gray')
            axes[row, 0].set_title(f'Original\nLabel: {ex["label"]}')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(ex['degraded'], cmap='gray')
            axes[row, 1].set_title(f'{dtype.upper()}\n(degraded)')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(ex['restored'], cmap='gray')
            axes[row, 2].set_title(f'{dtype.upper()}\n(restored)')
            axes[row, 2].axis('off')

            # Show OCR results with effect indicator
            before_correct = ex['before'] == ex['label']
            after_correct = ex['after'] == ex['label']
            before_color = 'green' if before_correct else 'red'
            after_color = 'green' if after_correct else 'red'

            # Determine effect of morphology
            if after_correct and not before_correct:
                effect = "IMPROVED!"
                effect_color = 'green'
            elif before_correct and not after_correct:
                effect = "WORSE!"
                effect_color = 'red'
            elif ex.get('changed', False):
                effect = "Changed"
                effect_color = 'orange'
            else:
                effect = "No change"
                effect_color = 'gray'

            axes[row, 3].text(0.5, 0.7, f"OCR Before: '{ex['before']}'",
                             ha='center', va='center', fontsize=14, color=before_color,
                             transform=axes[row, 3].transAxes)
            axes[row, 3].text(0.5, 0.5, f"OCR After: '{ex['after']}'",
                             ha='center', va='center', fontsize=14, color=after_color,
                             transform=axes[row, 3].transAxes)
            axes[row, 3].text(0.5, 0.3, f"Ground Truth: {ex['label']}",
                             ha='center', va='center', fontsize=12, color='black',
                             transform=axes[row, 3].transAxes)
            axes[row, 3].text(0.5, 0.1, f"-> {effect}",
                             ha='center', va='center', fontsize=14, fontweight='bold',
                             color=effect_color, transform=axes[row, 3].transAxes)
            axes[row, 3].axis('off')

        plt.suptitle('Pipeline Evaluation Examples\n(Green = correct, Red = incorrect)', fontsize=14)
        plt.tight_layout()
        plt.show()

    # Clean up
    del results['counts']

    return results


def create_morphological_challenge(difficulty='medium', seed=42):
    """
    Create a set of degraded SVHN digits as a challenge for students.

    Returns digits with various degradations that students must restore
    using their morphological pipeline.

    Args:
        difficulty: 'easy', 'medium', or 'hard'
            - easy: Light degradation, clear digits
            - medium: Moderate degradation, some ambiguity
            - hard: Heavy degradation, challenging cases
        seed: Random seed for reproducibility

    Returns:
        dict: Challenge dataset with keys:
            - 'images': List of degraded binary images
            - 'labels': Ground truth labels
            - 'degradation_types': Type of degradation for each image
            - 'originals': Original binary images (for comparison)
    """
    # Load SVHN data
    n_samples = 10
    images, labels = load_svhn_cropped('data/test_32x32.mat', n_samples=n_samples, seed=seed)

    # Prepare binary versions
    binary_digits = [prepare_svhn_for_morphology(img, target_size=(100, 100)) for img in images]

    # Set degradation intensity by difficulty
    if difficulty == 'easy':
        salt_ratio, pepper_ratio = 0.01, 0.01
        erosion_kernel, erosion_iter = 2, 1
    elif difficulty == 'medium':
        salt_ratio, pepper_ratio = 0.02, 0.02
        erosion_kernel, erosion_iter = 2, 1
    else:  # hard
        salt_ratio, pepper_ratio = 0.03, 0.03
        erosion_kernel, erosion_iter = 3, 1

    degradation_types = ['salt', 'pepper', 'eroded', 'combined'] * (n_samples // 4 + 1)
    degradation_types = degradation_types[:n_samples]
    np.random.seed(seed)
    np.random.shuffle(degradation_types)

    degraded_images = []
    for img, dtype in zip(binary_digits, degradation_types):
        if dtype == 'salt':
            degraded = add_salt_noise(img, salt_ratio)
        elif dtype == 'pepper':
            degraded = add_pepper_noise(img, pepper_ratio)
        elif dtype == 'eroded':
            degraded = apply_stroke_erosion(img, erosion_kernel, erosion_iter)
        else:  # combined
            eroded = apply_stroke_erosion(img, erosion_kernel, erosion_iter)
            with_pepper = add_pepper_noise(eroded, pepper_ratio * 0.5)
            degraded = add_salt_noise(with_pepper, salt_ratio * 0.5)
        degraded_images.append(degraded)

    return {
        'images': degraded_images,
        'labels': labels,
        'degradation_types': degradation_types,
        'originals': binary_digits
    }


# =============================================================================
# Region Growing Segmentation and Visualization
# =============================================================================

from collections import deque
import imageio
from IPython.display import Image, display


def get_neighbors(image, p):
    """
    Get 4-connected neighbors of pixel p within image bounds.

    Args:
        image: Input image (used to check bounds)
        p: Tuple (row, col) of pixel coordinates

    Returns:
        list: List of valid neighbor coordinates
    """
    neighbors = []
    for delta in [(0, -1), (-1, 0), (1, 0), (0, 1)]:
        n = (p[0] + delta[0], p[1] + delta[1])
        if 0 <= n[0] < image.shape[0] and 0 <= n[1] < image.shape[1]:
            neighbors.append(n)
    return neighbors


def region_growing_generator(image, seed_points, similarity_criterion):
    """
    Region growing segmentation generator.

    Yields intermediate segmentation states for visualization.

    Args:
        image: Input grayscale image (normalized to [0,1])
        seed_points: Tuple of (row, col) seed coordinates
        similarity_criterion: Function(pixel_value, region_id, label_array) -> bool

    Yields:
        np.ndarray: Label array with region IDs (-1 for unassigned)
    """
    result = -1 * np.ones(image.shape, dtype=int)
    region_id = 0
    queue = deque()

    for s in seed_points:
        result[s] = region_id
        queue.append((s, region_id))
        region_id += 1

    while queue:
        p, rid = queue.popleft()
        for n in get_neighbors(image, p):
            if result[n] == -1 and similarity_criterion(image[n], rid, result):
                result[n] = rid
                queue.append((n, rid))
        yield result


def make_similarity_criterion(img, tau=0.15):
    """
    Factory function to create a similarity criterion bound to a specific image.

    The criterion compares pixel intensity to the mean of the current region.

    Args:
        img: The image to use for computing region means
        tau: Similarity threshold (default 0.15)

    Returns:
        Function compatible with region_growing_generator
    """
    def criterio_similaridade(pixel_value, region_id, L):
        media_i = np.mean(img[L == region_id])
        return np.abs(pixel_value - media_i) <= tau
    return criterio_similaridade


def _render_segmentation_frame(image, segmentation, seed_points, cmap, show_seeds, iteration, title_suffix=""):
    """
    Render a segmentation frame as numpy array for video/GIF.

    Args:
        image: Original grayscale image
        segmentation: Label array from region growing
        seed_points: Tuple of (row, col) seed coordinates
        cmap: Colormap for segmentation visualization
        show_seeds: Whether to plot seed points
        iteration: Current iteration number
        title_suffix: Additional text for title

    Returns:
        np.ndarray: RGB numpy array suitable for video frame
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    if show_seeds:
        for s in seed_points:
            axes[0].plot(s[1], s[0], 'ro', markersize=5)
    axes[0].axis('off')

    axes[1].imshow(segmentation, cmap=cmap)
    title = f"Segmentation (iter: {iteration})"
    if title_suffix:
        title += f" {title_suffix}"
    axes[1].set_title(title)
    axes[1].axis('off')

    plt.tight_layout()

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = frame[:, :, :3]

    plt.close(fig)
    return frame


def visualize_region_growing(
    image,
    seed_points,
    similarity_criterion,
    morphological_ops=None,
    apply_morph_per_frame=False,
    frame_step=500,
    output_path="region_growing.gif",
    fps=10,
    show_seeds=True,
    cmap='nipy_spectral',
    display_in_notebook=True,
    max_iterations=None,
    verbose=True
):
    """
    Visualize region growing segmentation as an animated GIF.

    Args:
        image: Input grayscale image (np.ndarray, normalized to [0,1])
        seed_points: Tuple of (row, col) seed coordinates
        similarity_criterion: Function for region membership
        morphological_ops: List of callables for post-processing (optional)
        apply_morph_per_frame: If True, apply ops to each frame
        frame_step: Sample every N iterations (default 500)
        output_path: GIF output filename (default "region_growing.gif")
        fps: Frames per second (default 10)
        show_seeds: Show seed points on original image (default True)
        cmap: Colormap for segmentation (default 'nipy_spectral')
        display_in_notebook: Display GIF in notebook (default True)
        max_iterations: Maximum iterations (default None = all pixels)
        verbose: Print progress (default True)

    Returns:
        np.ndarray: Final segmentation result
    """
    frames = []
    final_result = None

    generator = region_growing_generator(image, seed_points, similarity_criterion)

    total_pixels = image.shape[0] * image.shape[1]
    if max_iterations is None:
        max_iterations = total_pixels

    for i, result in enumerate(generator):
        final_result = result.copy()

        if i % frame_step == 0:
            if verbose:
                assigned = np.sum(result >= 0)
                print(f"Iteration {i}: {assigned}/{total_pixels} pixels ({100*assigned/total_pixels:.1f}%)")

            seg_to_render = result.copy()

            if apply_morph_per_frame and morphological_ops is not None:
                for op in morphological_ops:
                    seg_to_render = op(seg_to_render)

            frame = _render_segmentation_frame(image, seg_to_render, seed_points, cmap, show_seeds, i)
            frames.append(frame)

        if i >= max_iterations:
            if verbose:
                print(f"Stopped at max_iterations={max_iterations}")
            break

    if verbose:
        print(f"Region growing completed at iteration {i}")

    frame = _render_segmentation_frame(image, final_result, seed_points, cmap, show_seeds, i, "(final)")
    frames.append(frame)

    if not apply_morph_per_frame and morphological_ops is not None:
        post_processed = final_result.copy()
        for idx, op in enumerate(morphological_ops):
            post_processed = op(post_processed)
            frame = _render_segmentation_frame(
                image, post_processed, seed_points, cmap, show_seeds,
                i, f"(morph op {idx + 1})"
            )
            frames.append(frame)
        final_result = post_processed

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path} ({len(frames)} frames)")

    if display_in_notebook:
        display(Image(filename=output_path))

    return final_result


def compute_region_areas(segmentation, pixel_area_mm2=1.0):
    """
    Compute the area of each labeled region in physical units.

    Args:
        segmentation: Label array from region growing (-1 = unassigned)
        pixel_area_mm2: Area of each pixel in mm^2 (default 1.0)

    Returns:
        dict: {region_id: area_in_mm2} for each region
    """
    areas = {}
    unique_labels = np.unique(segmentation)

    for label in unique_labels:
        if label >= 0:
            pixel_count = np.sum(segmentation == label)
            areas[label] = pixel_count * pixel_area_mm2

    return areas


def merge_regions(segmentation, labels_to_merge, new_label=None):
    """
    Merge multiple region labels into a single region.

    Args:
        segmentation: Label array from region growing
        labels_to_merge: List of region IDs to merge (e.g., [0, 2])
        new_label: Label for merged region (default: min of labels_to_merge)

    Returns:
        np.ndarray: Updated segmentation with merged regions
    """
    result = segmentation.copy()

    if new_label is None:
        new_label = min(labels_to_merge)

    for label in labels_to_merge:
        result[segmentation == label] = new_label

    return result


def compute_total_segmented_area(segmentation, pixel_area_mm2=1.0):
    """
    Compute total area of all segmented regions.

    Args:
        segmentation: Label array from region growing (-1 = unassigned)
        pixel_area_mm2: Area of each pixel in mm^2

    Returns:
        float: Total segmented area in mm^2
    """
    return np.sum(segmentation >= 0) * pixel_area_mm2
