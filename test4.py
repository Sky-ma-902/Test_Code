import cv2
from pyzbar.pyzbar import decode


def Change_To_codes(image_path):
    image=cv2.imread(image_path)
    decoded_objects=decode(image)
    for obj in decoded_objects:
        Qr_data=obj.data.decode('utf-8')
        Qr_type=obj.type
        points=obj.polygon
        if len(points)==4:
            pts=[(point.x,point.y)for point in points]
            for j in range(4):
                cv2.line(image,pts[j],pts[(j+1)%4],(0,255,0),3)
            cv2.putText(image,Qr_data,(pts[0][0],pts[0][1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            print(f"二维码: {Qr_type}")
            print(f"二维码内: {Qr_data}")
            print(f"顶点坐标: {pts}")
    cv2.imshow('QR CODE DECODE',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    # 替换为你的图像文件路�??
    image_path = './gragh/2.png'
    Change_To_codes(image_path)