import cv2
import threshold_decomposition as td

image = cv2.imread("./sample_images/sat_map3.jpg")

red = image[:, :, 2]
green = image[:, :, 1]
blue = image[:, :, 0]

cv2.imshow("main", green)

cv2.waitKey(0)
cv2.destroyAllWindows()
