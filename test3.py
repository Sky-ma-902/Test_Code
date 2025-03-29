import cv2
from pyzbar import pyzbar
def safe_decode(gray):
    try:
        return pyzbar.decode(gray)
    except Exception as e:
        print(f"error if:{str(e)}")

def detect(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 识别二维码
    barcodes = safe_decode(gray)
    
    # 遍历所有检测到的二维码
    for barcode in barcodes:
        # 提取二维码内容的字节数据，转换为字符串
        data = barcode.data.decode("utf-8")
        print(f"二维码内容: {data}")
        
        # 获取顶点坐标
        points = barcode.polygon
        print("顶点坐标:", [(p.x, p.y) for p in points])
        
        # 如果检测到四边形（常规二维码）
        if len(points) == 4:
            # 将坐标转换为tuple格式
            position = [(p.x, p.y) for p in points]
            
            # 在图像上绘制边框
            for i in range(4):
                cv2.line(image, position[i], position[(i+1)%4], (0, 255, 0), 2)
            
            # 绘制顶点坐标点
            for (x, y) in position:
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            
            # 显示二维码内容文本
            cv2.putText(image, data, (position[0][0], position[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 显示结果
    cv2.imshow("Code content", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "./gragh/3.jpg"  
    detect(image_path)