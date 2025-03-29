from ultralytics import YOLO

# 加载模型（直接加载预训练权重即可）
model = YOLO("yolo11n-cls.pt")  

# 训练模型
model.train(data="mnist160", epochs=100, imgsz=64)

# 使用训练好的模型进行预测
result = model("./bus.jpg")  