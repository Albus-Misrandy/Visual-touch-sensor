import cv2

# 打开两个摄像头
cap0 = cv2.VideoCapture(0)  # 摄像头0
cap1 = cv2.VideoCapture(1)  # 摄像头1

# 定义编码器
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out0 = None  # 录像输出0
out1 = None  # 录像输出1
recording = False  # 是否正在录制

while cap0.isOpened() and cap1.isOpened():
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not (ret0 and ret1):
        print("某个摄像头断开！")
        break

    # 显示两个摄像头画面
    cv2.imshow('Camera 0', frame0)
    cv2.imshow('Camera 1', frame1)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and not recording:
        print("开始录制两个摄像头...")
        out0 = cv2.VideoWriter('output_cam0.avi', fourcc, 20.0, (frame0.shape[1], frame0.shape[0]))
        out1 = cv2.VideoWriter('output_cam1.avi', fourcc, 20.0, (frame1.shape[1], frame1.shape[0]))
        recording = True

    if recording:
        out0.write(frame0)
        out1.write(frame1)

    if key == ord('q'):
        print("录制结束，保存文件。")
        break

# 释放所有资源
cap0.release()
cap1.release()
if out0:
    out0.release()
if out1:
    out1.release()
cv2.destroyAllWindows()
