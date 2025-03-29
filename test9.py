import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np

def classify_image(model, image_path):
    # 加载图像
    image = plt.imread(image_path)
    
    # 执行预测
    results = model.predict(image, conf=0.5)
    
    # 获取最高概率的类别索引
    class_idx = results[0].probs.top1
    
    # 可视化结果
    plt.imshow(image)
    plt.title(f"预测类别：{results[0].names[class_idx]}\n \
              置信度：{float(np.array(results[0].probs[class_idx]) )* 100:.2f}%")
    plt.axis('off')
    plt.show()

# 使用示例（需加载分类模型）
model = YOLO("yolo11n-cls.pt")  # 确保使用分类专用模型[1,3](@ref)
classify_image(model, "./gragh/1.jpg")