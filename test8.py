import cv2
import numpy as np
def process(img):
    #CLAHE 直方图均衡化（应对光照不均）
    LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    L = clahe.apply(L)
    LAB = cv2.merge((L,A,B))
    img = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
    return img

def dynamicrange(img):
    #动态阈值调整
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    avgv = np.mean(v)
    
    # 根据平均亮度调整阈值
    if avgv < 80:   # 弱光环境
        imgv_adjust = -30
        imgs_adjust = -20
    elif avgv > 180: # 强光环境
        imgv_adjust = +50
        imgs_adjust = +20
    else:            # 正常光照
        imgv_adjust = 0
        imgs_adjust = 0
    
    color_range = {
        
        'red': 
        {
            'rgb': ([(150, 255), (0, 120), (0, 120)]),
            'hsv': [(0, max(50+imgs_adjust,0), max(50+imgv_adjust,0), 15, 255, 255),
                (165, max(50+imgs_adjust,0), max(50+imgv_adjust,0), 180, 255, 255)]
        },
        'blue': 
        {
            'rgb': ([(0, 120), (0, 120), (150, 255)]),
            'hsv': [(85, max(50+imgs_adjust,0), max(50+imgv_adjust,0), 130, 255, 255)]
        },
        'yellow': 
        {
            'rgb': ([(180, 255), (180, 255), (0, 120)]),
            'hsv': [(20, max(50+imgs_adjust,0), max(50+imgv_adjust,0), 40, 255, 255)]
        },
        'green': 
        {
            'rgb': ([(0, 120), (100, 255), (0, 120)]),
            'hsv': [(40, max(50+imgs_adjust,0), max(50+imgv_adjust,0), 90, 255, 255)]
        },
        'black': 
        {
            'rgb': ([(0, 50+imgv_adjust), (0, 50+imgv_adjust), (0, 50+imgv_adjust)]),
            'hsv': [(0, 0, 0, 180, 255, 30+imgv_adjust)]
        },
        'white': 
        {
            'rgb': ([(200, 255), (200, 255), (200, 255)]),
            'hsv': [(0, 0, max(200+imgv_adjust,0), 180, 30-imgs_adjust, 255)]
        }        
    }
    return color_range
def detect_color(image_path):
    #读取图像
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
    else:
        img = process(img)
    height, width = img.shape[:2]
    total_pixels = height * width
    
    #转换颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #将RGB图像分解为三个单通道矩阵
    r, g, b = cv2.split(rgb_img)
    def RGB_cover(color):
        color_set = dynamicrange(img)
        ranges = color_set[color]['rgb']
        cover_r = cv2.inRange(r, ranges[0][0], ranges[0][1])
        cover_g = cv2.inRange(g, ranges[1][0], ranges[1][1])
        cover_b = cv2.inRange(b, ranges[2][0], ranges[2][1])
        cover = cv2.bitwise_and(cover_r, cv2.bitwise_and(cover_g, cover_b))
        #生成掩膜后进行去噪处理
        k1 = np.ones((5,5),np.uint8)
        cover = cv2.morphologyEx(cover,cv2.MORPH_CLOSE,k1)
        return cover
    
    #生成相应颜色的掩膜
    def HSV_cover(color):
        cover = np.zeros((height, width), dtype=np.uint8)
        color_set = dynamicrange(img)
        for range in color_set[color]['hsv']:
            lower = np.array([range[0], range[1], range[2]])
            upper = np.array([range[3], range[4], range[5]])
            cover = cv2.bitwise_or(cover, cv2.inRange(hsv_img, lower, upper))
            k1 = np.ones((5,5),np.uint8)
            cover = cv2.morphologyEx(cover,cv2.MORPH_CLOSE,k1)
        return cover

    # RGB检测
    max_rgb = {'count': 0, 'color': 'unknown'}
    color_set = dynamicrange(img)
    for color in color_set:
        cover = RGB_cover(color)
        count = cv2.countNonZero(cover)
        if count > max_rgb['count']:
            max_rgb = {'count': count, 'color': color}

    # HSV检测
    max_hsv = {'count': 0, 'color': 'unknown'}
    color_set = dynamicrange(img)
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

image_path = './gragh/stronglight_images/0_cropped.jpg'
RGB_RESULT, HSV_RESULT = detect_color(image_path)
print(f"RGB检测结果: {RGB_RESULT}")
print(f"HSV检测结果: {HSV_RESULT}")