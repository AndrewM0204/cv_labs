import time

import cv2
import numpy as np


def native_erosion(image_2d_list):
    """
    Нативная реализация эрозии на Python.
    image_2d_list: двумерный список значений пикселей (0 или 255).
    """
    h = len(image_2d_list)
    w = len(image_2d_list[0])
    result = [[0] * w for _ in range(h)]
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (image_2d_list[i-1][j-1] == 255 and image_2d_list[i-1][j] == 255 and image_2d_list[i-1][j+1] == 255 and
                image_2d_list[i][j-1]   == 255 and image_2d_list[i][j]   == 255 and image_2d_list[i][j+1]   == 255 and
                image_2d_list[i+1][j-1] == 255 and image_2d_list[i+1][j] == 255 and image_2d_list[i+1][j+1] == 255):
                result[i][j] = 255
            else:
                result[i][j] = 0
                
    return result

def main():
    img_path = "images/test_image.png"

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    h, w = binary_img.shape

    img_list = binary_img.tolist()
    
    start_time_native = time.time()
    native_result_list = native_erosion(img_list)
    end_time_native = time.time()
    
    native_result = np.array(native_result_list, dtype=np.uint8)
    time_native = end_time_native - start_time_native

    kernel = np.ones((3, 3), np.uint8)
    
    start_time_cv = time.time()
    cv2_result = cv2.erode(binary_img, kernel, iterations=1)
    end_time_cv = time.time()
    
    time_cv = end_time_cv - start_time_cv


    print(f"Размер изображения: {w} x {h}")
    print(f"Время выполнения (Нативный Python): {time_native:.6f} сек.")
    print(f"Время выполнения (OpenCV):          {time_cv:.6f} сек.")
    print(f"Ускорение при использовании OpenCV: ~{int(time_native / max(time_cv, 1e-6))} раз")

    cv2.imwrite("images/result_native.png", native_result)
    cv2.imwrite("images/result_opencv.png", cv2_result)
    cv2.imwrite("images/original_binary.png", binary_img)

if __name__ == "__main__":
    main()