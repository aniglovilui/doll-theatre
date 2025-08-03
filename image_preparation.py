from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import os



def sobelOperator(object_array_64f):
    """
    Кастомная реализация оператора Собеля

    Args:
        object_array_64f: Входной массив float64 значений пикселей (оттенки серого)
    """
    
    # Ядра Собеля
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) 
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) 

    # Вычисление свертки
    rows, cols = object_array_64f.shape
    gradient_x = np.zeros_like(object_array_64f)
    gradient_y = np.zeros_like(object_array_64f)

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            gradient_x[i, j] = np.sum(object_array_64f[i - 1:i + 2, j - 1:j + 2] * sobel_x)
            gradient_y[i, j] = np.sum(object_array_64f[i - 1:i + 2, j - 1:j + 2] * sobel_y)

    # Вычисляем градиент и применяем пороговую обработку
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = np.clip(gradient, 0, 255).astype(np.uint8) # Обрезаем значения от 0 до 255
    
    return gradient # Возвращает объект массив np.array(uint8)



def prepareObjectImage(input_image_path, threshold_flood=50, need_castom_Sobel=False, need_save=False, output_path=None):
    """
    Подготовка изображения объекта

    Реализация алгоритма основана на статье [Простой фильтр для автоматического удаления фона с изображений] автора ArXen42 
    Оригинал: https://habr.com/ru/articles/353890/  
    
    Args:
        input_image_path: Путь к входному изображению
        threshold_flood: Пороговое значение заливки
        output_path: Путь для сохранения результата
        need_save: Булевый параметр, сигнализирующий о необходимости сохранять изображение
    """
    try:
        image = cv2.imread(input_image_path)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if need_castom_Sobel:
            # Кастомный оператор Собеля
            blurred_array = gaussian_filter(np.array(image_grey, dtype=np.float64), sigma=1.5) # sigma контролирует степень размытия
            magnitude = sobelOperator(blurred_array) 
        else:
            # Применяем оператор собеля
            gradient_x = cv2.Sobel(image_grey, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(image_grey, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.zeros_like(gradient_x, dtype=np.float64)
            cv2.magnitude(gradient_x, gradient_y, magnitude)

        magnitude += 50 # Прибавляем константу для корректной заливки
        magnitude = cv2.convertScaleAbs(magnitude) # Преобразуем в uint8 для отображения
        alpha_mask = magnitude.copy()

        element_size = int(round(np.sqrt(alpha_mask.shape[0] * alpha_mask.shape[1]) / 300))  # Округление
        element_size = element_size if element_size % 2 != 0 else element_size + 1  # Гарантируем нечетность
        element_size = max(3, min(element_size, 20)) # Ограничение размера
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))

        if need_castom_Sobel:
            cv2.morphologyEx(alpha_mask, cv2.MORPH_DILATE, ellipse_kernel, alpha_mask)
            cv2.morphologyEx(alpha_mask, cv2.MORPH_ERODE, ellipse_kernel, alpha_mask) # Компенсация начальной дилатации
      
        # Заливка внешней области
        # Результат зависит от значений loDiff, upDiff
        mask = np.zeros((alpha_mask.shape[0]+2, alpha_mask.shape[1]+2), np.uint8)
        seed_point1 = (0, 0) # Выбираем начальную точку для заливки (верхний левый угол)
        seed_point2 = (alpha_mask.shape[1]-1, 0) # Выбираем начальную точку для заливки (верхний правый угол)
        fill_color = 0
        cv2.floodFill(alpha_mask, mask, seed_point1, fill_color, 
                      loDiff=threshold_flood, upDiff=threshold_flood, 
                      flags=cv2.FLOODFILL_FIXED_RANGE) 
        cv2.floodFill(alpha_mask, mask, seed_point2, fill_color, 
                      loDiff=threshold_flood, upDiff=threshold_flood, 
                      flags=cv2.FLOODFILL_FIXED_RANGE)

        # cv2.morphologyEx(alpha_mask, cv2.MORPH_DILATE, ellipse_kernel, alpha_mask)
        cv2.morphologyEx(alpha_mask, cv2.MORPH_ERODE, ellipse_kernel, alpha_mask) # Компенсация начальной дилатации
        cv2.morphologyEx(alpha_mask, cv2.MORPH_OPEN, ellipse_kernel, alpha_mask, iterations=2) # Удаление шума
        cv2.morphologyEx(alpha_mask, cv2.MORPH_ERODE, ellipse_kernel, alpha_mask, iterations=2) # Корректировка контуров
        
        _, alpha_mask = cv2.threshold(alpha_mask, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(image)
        array_without_background = cv2.merge((b, g, r, alpha_mask))

        # Преобразование порядка каналов: BGR -> RGB
        result_array_rgba = cv2.cvtColor(array_without_background, cv2.COLOR_BGRA2RGBA)
        # Преобразование NumPy массива в PIL Image объект
        result_image_for_PIL = Image.fromarray(result_array_rgba, "RGBA")
        # Обрезка пустых краев изображения
        trimmed_result_image = result_image_for_PIL.crop(result_image_for_PIL.getbbox())

        trimmed_result_image = Image.fromarray(alpha_mask, "L") #!!!!!!!!!!!!!
        
        # Сохраняем результат
        if need_save:
            if output_path == None:
                # Получаем путь к файлу без расширения
                path_without_extension = os.path.splitext(input_image_path)[0]
                # Получаем расширение файла
                extension = os.path.splitext(input_image_path)[1]
                output_path = path_without_extension + "_prepared" + extension
            trimmed_result_image.save(output_path)
            print(f"prepareObjectImage -> Обрезанное изображение объекта без фона сохранено: {output_path}")

        return trimmed_result_image
    
    except FileNotFoundError:
        print(f"prepareObjectImage -> Файл не найден: {input_image_path}")
    except Exception as e:
        print(f"prepareObjectImage -> Произошла ошибка: {e}")



def affineTransform(object_array, angle_in_degrees=0, tx=0, ty=0, sx=1, sy=1, mirror=False):
    """
    Выполняет аффинное преобразование (поворот, перемещение и масштабирование) изображения

    Args:
        object_array: Входное изображение как NumPy массив
        angle_in_degrees: Угол поворота в градусах (положительный — против часовой стрелки)
        tx: Перемещение по оси x
        ty: Перемещение по оси y
        sx: Масштаб по оси x
        sy: Масштаб по оси y
    """
    h, w, c = object_array.shape
    center_x, center_y = w // 2, h // 2
    angle_in_radians = np.radians(angle_in_degrees)

    # Матрица поворота
    R = np.array([
        [np.cos(angle_in_radians), -np.sin(angle_in_radians), 0],
        [np.sin(angle_in_radians), np.cos(angle_in_radians), 0],
        [0, 0, 1]
        ])

    # Матрица масштабирования
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1] ])

    # Матрицы перемещения
    T = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [center_x+tx, center_y+ty, 1] ])
    T_ = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [-center_x, -center_y, 1] ])

    # Объединенная матрица преобразования
    M = T_ @ R @ T @ S

    # Создаем выходное изображение
    D = np.sqrt(h**2 + w**2)
    delta_h = int(D - h) + 1
    delta_w = int(D - w) + 1
    h1 = int( (h + delta_h)*sy + np.abs(ty) )+1
    w1 = int( (w + delta_w)*sx + np.abs(tx) )+1
    transformed_image = np.zeros((h1, w1, c), dtype=object_array.dtype)
    for y in range(h):
        for x in range(w):
            original_coords = np.array([x, y, 1])

            transformed_coords = original_coords @ M

            # Проверка на выход за границы
            tx_coord = int(transformed_coords[0]) + int(delta_w*sx/2)
            ty_coord = int(transformed_coords[1]) + int(delta_h*sy/2)
            if 0 <= tx_coord < w1 and 0 <= ty_coord < h1:
                transformed_image[ty_coord, tx_coord] = object_array[y, x]

    # Преобразование NumPy массива в PIL Image объект
    transformed_image_for_PIL = Image.fromarray(transformed_image, "RGBA")   
    # Обрезка пустых краев изображения
    trimmed_transformed_image_for_PIL = transformed_image_for_PIL.crop(transformed_image_for_PIL.getbbox())
    transformed_image = np.asarray(trimmed_transformed_image_for_PIL)
    # Зеркальное отображение
    if mirror: transformed_image = np.flip(transformed_image, axis=1) 
    return transformed_image # Возвращает NumPy массив
    