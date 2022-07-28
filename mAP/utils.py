import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Patch
from ipywidgets import interact

##img 불러오기
def img_read(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img

def _int2bool(item): 
    bool_list = [bool(i) for i in item]
    return np.array(bool_list)

##sample 보여주기
def show_sample(coco_tr , coco_de , class_name , index=0):
    legend_elements = [Patch(facecolor=(1.0 , 0, 0), edgecolor=(0,0,0),
                             label='True'),
                          Patch(facecolor=(1.0 , 1.0 , 0), edgecolor=(0,0,0),
                             label='Pred')]
    plt.figure(figsize = (13 , 15))
    plt.legend( handles = legend_elements, loc = 'upper left', fontsize=20 )
    data_dir = 'image/'
    
    
    file_name = coco_tr.loadImgs(ids = index + 1)[0]['file_name']
    anns_ids = coco_tr.getAnnIds(imgIds = [index + 1 ])
    anns = coco_tr.loadAnns(anns_ids)
    img = img_read(data_dir + file_name)
    
    for ann in anns:
        cv2.rectangle(img , ann['bbox'][:2] , ann['bbox'][2:] , (255 , 0 , 0) ,  4)
        cv2.putText(img , f"{class_name[str(ann['category_id'])]}" ,
                    org = ann['bbox'][:2] , fontFace = 1 , thickness = 3 , 
                    fontScale = 2 , color =(255, 0,  0)  )
        
    anns_de = coco_de.loadAnns(anns_ids)
    for ann in anns_de:
        cv2.rectangle(img , ann['bbox'][:2] , ann['bbox'][2:] , (255 , 255 , 0) ,  4)
        cv2.putText(img , f"{class_name[str(ann['category_id'])], ann['confidence_score']}" ,
                    org = ann['bbox'][:2] , fontFace = 1 , thickness = 3 , 
                    fontScale = 2 , color =(255, 255,  0)  )
    plt.title(file_name)
    plt.imshow(img)
    plt.show()

######  IOU 구하기 #################

## bbox의 영역 넓이 구하기
def _get_Area(bbox):
    xmin , ymin , xmax , ymax = bbox 
    area = (xmax - xmin + 1) * (ymax - ymin + 1)
    return area

## 두 박스의 교집합 영역 구하기
def _get_InterSectionArea(tr_bbox, de_bbox):
    xA = max(tr_bbox[0], de_bbox[0])
    yA = max(tr_bbox[1], de_bbox[1])
    xB = min(tr_bbox[2], de_bbox[2])
    yB = min(tr_bbox[3], de_bbox[3])
    # intersection area
    intersectionarea = (xB - xA + 1) * (yB - yA + 1)
    return intersectionarea

## 합집합 영역 구하기
def _UnionArea(tr_bbox, de_bbox, interArea=None):
    area_A = _get_Area(tr_bbox)
    area_B = _get_Area(de_bbox)
    
    ##  겹치는 부분 없으면 그냥 교집합 영역 구하기
    if interArea is None:
        interArea = _get_InterSectionArea(tr_bbox, de_bbox)

    return float(area_A + area_B - interArea)

##box 겹치는 부분 있는지 확인
def _boxesIntersect(tr_bbox, de_bbox):
    if tr_bbox[0] > de_bbox[2]:
        return False  # tr_bbox is right of de_bbox
    if de_bbox[0] > tr_bbox[2]:
        return False  # tr_bbox is left of de_bbox
    if tr_bbox[3] < de_bbox[1]:
        return False  # tr_bbox is above de_bbox
    if tr_bbox[1] > de_bbox[3]:
        return False  # tr_bbox is below de_bbox
    return True

def IOU(tr_bbox, de_bbox):
    # if boxes dont intersect
    if _boxesIntersect(tr_bbox, de_bbox) is False:
        return 0
    interArea = _get_InterSectionArea(tr_bbox, de_bbox)
    union = _UnionArea(tr_bbox, de_bbox, interArea=interArea)

    # intersection over union
    result = interArea / union
    assert result >= 0
    return result

########## IOU 구하기 end ###################

######## AP 구하기 #####################

##confidence score 이상값 반환
class get_AP():
    def __init__(self , coco_de , coco_tr):
        self.coco_de = coco_de
        self.coco_tr = coco_tr

    def _get_over_score(self , cat_i , confidence_score ):
        all_pred_bbox = self.coco_de.getAnnIds(catIds = [cat_i])
        all_pred_info = self.coco_de.loadAnns(all_pred_bbox)
        confidence = np.array([[idx['id'] , idx['confidence_score']] for idx in all_pred_info])
        confidence = confidence[confidence[:, 1] >= confidence_score]
        pred_bbox = confidence[:, 0].astype(int).tolist()
        
        return pred_bbox


    def get_recall_precision(self  , cat_i , confidence_score  ):
        all_bbox = self.coco_tr.getAnnIds(catIds = [cat_i])
        all_pred_bbox = self._get_over_score(cat_i , confidence_score = confidence_score)
        all_bbox = {bbox : 0 for bbox in all_bbox} 
        
        ## image iterator
        all_img = self.coco_tr.getImgIds(catIds = [cat_i])
        for img_i in all_img:
            truth_anns = self.coco_tr.getAnnIds( imgIds = [img_i], catIds = [cat_i])
            
            ## 일정 confidence score 이상인 것만 뽑기
            pred_anns =  self.coco_de.getAnnIds( imgIds = [img_i] , catIds = [cat_i])
            pred_anns = list(set(pred_anns) & set(all_pred_bbox))
            # truth bbox 별로 iterator 돌기
            for truth_ann in truth_anns:
                iou_storage = []
                k = 0
                ## pred bbox 별로 iterator 돌기
                for pred_ann in pred_anns:
                    truth = self.coco_tr.loadAnns(truth_ann)[0]
                    pred = self.coco_de.loadAnns(pred_ann)[0]
                    iou = IOU(truth['bbox'] , pred['bbox'] )
                    if iou > 0.3:
                        iou_storage.append([pred['id'] , iou])
                        k += 1
                ## iou 0.3 이상인 것중에 가장 높은 값만 TP 변환
                if k>=1:
                    iou_storage = np.array(iou_storage)
                    idx = np.argmax(iou_storage[: , 1])
                    best_iou_id = iou_storage[idx , 0]
                    all_bbox[truth['id']] = int(best_iou_id)
        
        ## recall과 precision값 계산
        mask = _int2bool(all_bbox.values())
        if len(all_pred_bbox) == 0:
            recall = mask.sum() / len(all_bbox)
            precision = 1
            return recall , precision
        else:
            recall = mask.sum() / len(all_bbox)
            precision = mask.sum() / len(all_pred_bbox)
        return recall , precision

    ## 11보간법으로 recall , precision list 구하기
    def _get_AP_list(self , cat_i):
        AP_list = []
        for i in np.linspace(0, 1 ,11):
            recall , precision = self.get_recall_precision(cat_i =cat_i , confidence_score= i )
            AP_list.append([recall , precision])
        return AP_list

    def AP(self , cat_i):
        AP_list = self._get_AP_list(cat_i)
        df = pd.DataFrame(AP_list  , columns = ['recall' , 'precision']).groupby('recall').max()
        ## 구간 구해서 넓이 구하기
        result = (np.squeeze(df.values[1:]) *( (df.index.values[1:] - df.index.values[:-1]))).sum()
        return result