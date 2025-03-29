import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import os

def class_photo(model, img_path, output_path):
    # 统一使用OpenCV读取确保BGR格式
    img = cv2.imread(img_path)
    
    # 执行目标检测推理（关键修改）
    results = model.predict(img, conf=0.5)  # 使用predict方法并设置置信度阈值
    
    # 绘制检测结果
    annotated_img = results[0].plot()  # 自动绘制框和标签
    
    # 显示OpenCV处理后的图像
    cv2.imshow("检测结果", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存带标注的图像
    if output_path:
        cv2.imwrite(output_path, annotated_img)
        print(f"结果已保存至：{os.path.abspath(output_path)}")
    
    # # Matplotlib可视化（可选）
    # plt.figure(figsize=(12, 8))
    # plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    # plt.title("目标检测结果")
    # plt.axis('off')
    # plt.show()

# 必须使用目标检测模型（示例使用YOLOv8）
model = YOLO("yolo11n-cls.pt")  # 官方检测模型
# model = YOLO("yolov11n.pt")  # 如果使用v11需确认模型文件存在

class_photo(model, "./bus.jpg", "./output/detected_bus.jpg")