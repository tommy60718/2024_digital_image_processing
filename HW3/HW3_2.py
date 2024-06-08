import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_image():
    # Using a sample image from matplotlib
    image = plt.imread('Dog.jpg')[:, :, 0]  # Ensure the image path is correct
    return image

def fft_shift(D):
    """ Centering zero frequencies for visualization """
    return np.fft.fftshift(D)

def laplacian_filter(nrows, ncols):
    """ Generate a Laplacian kernel in frequency domain matching the image size """
    laplacian = np.zeros((nrows, ncols))
    center_x, center_y = nrows // 2, ncols // 2
    laplacian[center_x, center_y] = -4
    if center_x > 0:
        laplacian[center_x-1, center_y] = 1
        laplacian[center_x+1, center_y] = 1
    if center_y > 0:
        laplacian[center_x, center_y-1] = 1
        laplacian[center_x, center_y+1] = 1
    # Normalize the filter
    return laplacian / np.sum(np.abs(laplacian))

def main():
    image = load_image()
    image_fft = np.fft.fft2(image)
    image_fft_shifted = fft_shift(image_fft)

    nrows, ncols = image.shape
    laplacian = laplacian_filter(nrows, ncols)
    filtered_fft = image_fft_shifted * laplacian

    # Adjust the alpha to increase the sharpening effect
    alpha = 7  # Increase alpha for more pronounced sharpening
    combined_fft = image_fft_shifted + alpha * filtered_fft

    # Inverse FFT to convert back to spatial domain
    image_sharpened = np.fft.ifft2(np.fft.ifftshift(combined_fft)).real

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.abs(image_fft_shifted) + 1), cmap='gray')
    plt.title('Fourier Transform')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_sharpened, cmap='gray')
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
