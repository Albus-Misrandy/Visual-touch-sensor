import cv2


if __name__ == "__main__":
    img = cv2.imread("Image_Sampling/captured_images/0001.jpg")

    cv2.imshow("img", img)
    cv2.waitKey(0)
