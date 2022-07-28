import cv2

##img 불러오기
def img_read(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

## bbox_list show
def show_bboxes(img , bboxes):
    img_clone =  img.copy()
    for bbox in bboxes:
        cv2.rectangle(img_clone, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     color=(0, 255, 0), thickness=10)
    return img_clone