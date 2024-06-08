import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def convolution_2d(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))  # Flip the kernel
    output = np.zeros_like(image)  # Prepare the output array
    # Add zero padding to the image
    image_padded = np.pad(image, [(1, 1), (1, 1)], mode='constant', constant_values=0)

    for y in range(image.shape[0]):  # Loop over every pixel of the image
        for x in range(image.shape[1]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y+3, x:x+3]).sum()
            
    return output

def load_image(path):
    image = mpimg.imread(path)
    if image.dtype == np.uint8:
        image = image / 255.0  # Normalize to 0-1 if in uint8
    if len(image.shape) > 2 and image.shape[2] == 4:
        # Convert from RGBA to RGB
        image = image[..., :3]
    return image

def main():
    # Load an image
    image_path = 'test1.jpg'  # Update this path
    image = load_image(image_path)

    # Define a Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Apply the convolution function to the image
    if len(image.shape) == 3 and image.shape[2] == 3:  # Check if the image is colored
        image_filtered = np.zeros_like(image)
        for i in range(3):  # Apply filter to each channel
            image_filtered[:,:,i] = convolution_2d(image[:,:,i], laplacian_kernel)
    else:
        image_filtered = convolution_2d(image, laplacian_kernel)

    # Sharpen the image by adding the filtered image back to the original image
    sharpened_image = np.clip(image - image_filtered, 0, 1)

    # Show the original and the sharpened image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(sharpened_image, cmap='gray')
    plt.title('Sharpened Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
