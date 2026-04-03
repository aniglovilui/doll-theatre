import cv2
import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from math import ceil
from numpy.typing import NDArray



def sobelOperator(img_array: NDArray[np.float64]) -> NDArray[np.uint8]:
    """
    Castom realization of Sobel operator.

    Args:
        img_array: Input array of float64 pixel values (grayscale).

    Returns: 
        grad: Gradient matrix.
    """
    
    # sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) 

    # computing the convolution
    rows, cols = img_array.shape
    gradient_x = np.zeros_like(img_array)
    gradient_y = np.zeros_like(img_array)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            gradient_x[i, j] = np.sum(img_array[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            gradient_y[i, j] = np.sum(img_array[i - 1:i + 2, j - 1:j + 2] * sobel_y)

    # calculate the gradient and apply thresholding
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = np.clip(gradient, 0, 255).astype(np.uint8) 
    
    return gradient



def prepareObjectImage(
        input_image_path: str, 
        threshold_flood: int=50, 
        need_castom_sobel: bool=False, 
        relative_flood_seed_points: list[tuple[float, float]]=[(0, 0), (0.99, 0.99)],
        need_save: bool=False,
        output_path: str | None=None,
    ) -> Image.Image | None:
    """
    Подготовка изображения объекта

    The implementation of the algorithm for background deletion is based on the article [Простой фильтр для автоматического удаления фона с изображений] by ArXen42.
    Original: https://habr.com/ru/articles/353890/  
    
    Args:
        input_image_path: Путь к входному изображению
        threshold_flood: Пороговое значение заливки
        output_path: Путь для сохранения результата
        need_save: Булевый параметр, сигнализирующий о необходимости сохранять изображение

        input_image_path: Path to the input image;
        threshold_flood: Flood threshold;
        need_castom_sobel: Boolean parameter indicating whether to use castom Sobel operator;
        relative_flood_seed_points: List of relative coordinates of seedpoints for flooding;
        need_save: Boolean parameter indicating whether to save the result;
        output_path: Path to save the output.

    Returns: 
        trimmed_result_image: Resulting trimmed image with deleted background.
    """
    try:
        image = cv2.imread(input_image_path)
        if not np.any(image): 
            raise Exception(f"File not found: {input_image_path}.")
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if need_castom_sobel:
            # castom Sobel
            blurred_array = gaussian_filter(np.array(image_grey, dtype=np.float64), sigma=1.2) # sigma controls the degree of blur (1.2-1.5)
            magnitude = sobelOperator(blurred_array) 
        else:
            gradient_x = cv2.Sobel(image_grey, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image_grey, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.zeros_like(gradient_x, dtype=np.float64)
            cv2.magnitude(gradient_x, gradient_y, magnitude)

        magnitude += 50 # adding a constant for correct filling
        magnitude = cv2.convertScaleAbs(magnitude) # convert to uint8 for display
        alpha_mask = magnitude.copy()

        amask_h, amask_w = alpha_mask.shape
        element_size = int(round(np.sqrt(amask_h * amask_w) / 300))  # rounding
        element_size = element_size if element_size % 2 != 0 else element_size + 1  # to guarantee odd numbers
        element_size = max(3, min(element_size, 20)) # size limitation
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))

        if need_castom_sobel:
            # additional preprocessing to provide sharper contours
            cv2.morphologyEx(alpha_mask, cv2.MORPH_DILATE, ellipse_kernel, alpha_mask)
            cv2.morphologyEx(alpha_mask, cv2.MORPH_ERODE, ellipse_kernel, alpha_mask) 
      
        # filling the outer area (result very sensitive to loDiff and upDiff)
        mask = np.zeros((amask_h + 2, amask_w + 2), np.uint8)
        # starting points for filling (upper left corner and upper right corner, to enshure correct filling) 
        fill_color = 0
        for relative_seed in relative_flood_seed_points:
            seed = (
                ceil(relative_seed[0]*amask_w), 
                ceil(relative_seed[1]*amask_h)
            )
            cv2.floodFill(
                alpha_mask, 
                mask, 
                seed, 
                fill_color,
                loDiff=threshold_flood, 
                upDiff=threshold_flood, 
                flags=cv2.FLOODFILL_FIXED_RANGE,
                ) 
            
        cv2.morphologyEx(alpha_mask, cv2.MORPH_DILATE, ellipse_kernel, alpha_mask)
        cv2.morphologyEx(alpha_mask, cv2.MORPH_ERODE, ellipse_kernel, alpha_mask) # compensate initial dilate
        cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, ellipse_kernel, alpha_mask, iterations=3) # remove noise
        cv2.morphologyEx(alpha_mask, cv2.MORPH_ERODE, ellipse_kernel, alpha_mask, iterations=3) # contour correction
        
        _, alpha_mask = cv2.threshold(alpha_mask, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        array_without_background = cv2.merge((b, g, r, alpha_mask))

        # BGR -> RGB
        result_array_rgba = cv2.cvtColor(array_without_background, cv2.COLOR_BGRA2RGBA)
        # array -> image
        result_image_for_PIL = Image.fromarray(result_array_rgba, "RGBA")
        # crop blank sides of the image
        trimmed_result_image = result_image_for_PIL.crop(result_image_for_PIL.getbbox())
        
        # save result
        if need_save:
            if output_path == None:
                # Path without extension
                path_without_extension = os.path.splitext(input_image_path)[0]
                # Extension
                extension = os.path.splitext(input_image_path)[1]
                output_path = path_without_extension + "_prepared" + extension
            trimmed_result_image.save(output_path)
            print(f"prepareObjectImage -> Image without background is saved at: {output_path}")

        return trimmed_result_image
    
    except Exception as e:
        print(f"prepareObjectImage -> Error: {e}")