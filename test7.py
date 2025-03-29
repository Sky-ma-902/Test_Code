import cv2
import numpy as np

def detect_color(image_path):
    #读取图像
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
    
    height, width = img.shape[:2]
    total_pixels = height * width
    
    #转换颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #将RGB图像分解为三个单通道矩阵
    r, g, b = cv2.split(rgb_img)

    # 颜色阈值配置
    '''
    定义不同颜色分别在RGB与CSV空间中的颜色阈值范围
    RGB:[(Rmin,Rmax),(Gmin,Gmax),(Bmin,Bmax)]
    CSV:[(Cmin,Smin,Vmin),(Cmax,Smax,Vmax)]
    '''
    color_set = {
        'white': 
        {
            'rgb': ([(200, 255), (200, 255), (200, 255)]),
            'hsv': [(0, 0, 200, 180, 30, 255)]
        },
        'red': 
        {
            'rgb': ([(200, 255), (0, 100), (0, 100)]),
            'hsv': [(0, 50, 50, 10, 255, 255), (170, 50, 50, 180, 255, 255)]
        },
        'blue': 
        {
            'rgb': ([(0, 100), (0, 100), (200, 255)]),
            'hsv': [(85, 50, 50, 115, 255, 255)]
        },
        'yellow': 
        {
            'rgb': ([(200, 255), (200, 255), (0, 100)]),
            'hsv': [(20, 50, 50, 30, 255, 255)]
        },
        'green': 
        {
            'rgb': ([(0, 100), (100, 255), (0, 100)]),
            'hsv': [(30, 50, 50, 85, 255, 255)]
        },
        'black': {
            'rgb': ([(0, 30), (0, 30), (0, 30)]),
            'hsv': [(0, 0, 0, 180, 255, 30)]
        }
    }
    #生成相应颜色的掩膜
    def RGB_cover(color):
        ranges = color_set[color]['rgb']
        cover_r = cv2.inRange(r, ranges[0][0], ranges[0][1])
        cover_g = cv2.inRange(g, ranges[1][0], ranges[1][1])
        cover_b = cv2.inRange(b, ranges[2][0], ranges[2][1])
        cover = cv2.bitwise_and(cover_r, cv2.bitwise_and(cover_g, cover_b))
        #生成掩膜后进行去噪处理
        k1 = np.ones((5,5),np.unit8)
        cover = cv2.morphologyEx(cover,cv2.MORPH_CLOSE,k1)
        return cover
    
    #生成相应颜色的掩膜
    def HSV_cover(color):
        cover = np.zeros((height, width), dtype=np.uint8)
        for range in color_set[color]['hsv']:
            lower = np.array([range[0], range[1], range[2]])
            upper = np.array([range[3], range[4], range[5]])
            cover = cv2.bitwise_or(cover, cv2.inRange(hsv_img, lower, upper))
            k1 = np.ones((5,5),np.unit8)
            cover = cv2.morphologyEx(cover,cv2.MORPH_CLOSE,k1)
        return cover

    # RGB检测
    max_rgb = {'count': 0, 'color': 'unknown'}
    for color in color_set:
        cover = RGB_cover(color)
        count = cv2.countNonZero(cover)
        if count > max_rgb['count']:
            max_rgb = {'count': count, 'color': color}

    # HSV检测
    max_hsv = {'count': 0, 'color': 'unknown'}
    for color in color_set:
        cover = HSV_cover(color)
        count = cv2.countNonZero(cover)
        if count > max_hsv['count']:
            max_hsv = {'count': count, 'color': color}

    # 阈值判断（至少5%像素）
    min = total_pixels * 0.05
    RGB = max_rgb['color'] if max_rgb['count'] > min else 'unknown'
    HSV = max_hsv['color'] if max_hsv['count'] > min else 'unknown'

    return RGB, HSV


image_path = './gragh/normal_images/1_cropped.jpg'
RGB_RESULT, HSV_RESULT = detect_color(image_path)
print(f"RGB检测结果: {RGB_RESULT}")
print(f"HSV检测结果: {HSV_RESULT}")