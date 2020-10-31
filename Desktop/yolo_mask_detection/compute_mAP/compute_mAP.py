import xml.etree.ElementTree as ET
import os
import pickle as pk
import numpy as np
 
def parse_rec(filename):    #通過ET解析xml後返回一個obj
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)  #objects格式為 [{'name':egret,'pose':Unspecifie等},{'name':egret,'pose':Unspecifie等}]
 
    return objects
 
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
 
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
 
        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
 
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name 默認txt中是無後綴imgName
    # cachedir caches the annotations in a pickle file
 
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)  #若無pkl文件的路徑，生成cachedir路徑
    cachefile = os.path.join(cachedir, 'annots.pkl')    
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines] #imagenames為所有imgName的list
 
    if not os.path.isfile(cachefile):   #cache路徑下無pkl
        # load annots
        recs = {}   #recs是一個dict，以imagename為key，解析xml後的obj為value，詳情見下兩句
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename)) #依次寫入format上imagename的xml路徑到resc列表
            if i % 100 == 0:
                print ('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))) #   顯示進程
        # save
        print ('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pk.dump(recs, f)   #將resc列表中的內容寫入pkl
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pk.load(f)  #若存在pkl，直接load到recs   將‘str’轉化為'bytes'
 
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]    #除去recs中其他類別
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # read dets
    detfile = detpath.format(classname)
    print(detfile)
    # detfile = os.path.join(detpath, 'person.txt')
    with open(detfile, 'r') as f:#讀批量驗證的結果txt文件
        lines = f.readlines()
        
    splitlines = [x.strip().split(' ') for x in lines]  #split對txt每一行的數據做分割
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
 
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
 
    # go down dets and mark TPs and FPs 以下為計算對比各參數
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)
 
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih
 
            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
 
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
 
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
 
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
 
    return rec, prec, ap

# 分別計算每種label的AP，接著再計算mAP
file_names = [file for file in os.listdir('.') if 'yolov3_test_result_' in file]
mAP = 0
for file_name in file_names:
    class_name = os.path.splitext(file_name)[0].split('_')[-1]
    rec , prec , ap = voc_eval(os.path.join('.' , 'yolov3_test_result_{}.txt') ,
                               os.path.join('.' , '{}.xml') ,
                               os.path.join('.' , 'test_index.txt') ,
                               class_name , '.')
    print('class_AP_{} : {:.2f}'.format(class_name , ap))
    mAP += ap
    
mAP /= len(file_names) 
print('mAP : {:.3f}'.format(mAP))   
