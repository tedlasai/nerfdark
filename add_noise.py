

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('macosx')


def ungamma_correct(image, gamma=2.2):
    return np.power(image, gamma)


def gamma_correct(image, gamma=2.2):
    return np.power(image, 1 / gamma)


def apply_noise(image, slopes, intercepts):
    noise_variance = np.zeros_like(image, dtype=np.float32)
    for ch in range(3):
        noise_variance[..., ch] = slopes[ch] * image[..., ch] + intercepts[ch]
    noise_std = np.sqrt(np.maximum(noise_variance, 0))  # Ensure non-negative variance
    noise = np.random.normal(0, 1, image.shape) * noise_std
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image


def process_and_visualize(image_path):
    # Load image and normalize
    image = cv2.imread(image_path) / 255.0
    image = image[..., ::-1]



    # Noise parameters
    slopes = [0.17915034221637174, 0.07429921059808108, 0.09985825070217794]
    intercepts = [-0.0018851864039859886, 0.00011677127942324982, -0.001055012229532099]

    # Ungamma
    linear_image = ungamma_correct(image)

    # Apply noise
    noisy_linear_image = apply_noise(linear_image, slopes, intercepts)

    print(np.mean(np.abs(noisy_linear_image-linear_image)))

    # Gamma correction
    final_image = gamma_correct(noisy_linear_image)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(final_image)
    axes[1].set_title("Noisy Image")
    axes[1].axis("off")
    plt.show()


# Example usage
process_and_visualize("bear.jpg")
