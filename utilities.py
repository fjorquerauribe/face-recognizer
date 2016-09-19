import cv2
import matplotlib.pyplot as plt

def plt_show(image, title=""):
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.axis("off")
	plt.title(title)
	plt.imshow(image, cmap="Greys_r")
	plt.show()


def draw_rectangle_with_text(frame, top_left, bottom_right, message=""):
	cv2.rectangle(frame, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (150,150,0), 8)
	cv2.putText(frame, message, (top_left[0]-10,top_left[1]-10), cv2.FONT_HERSHEY_SIMPLEX, .7, (150,150,0), 2)
