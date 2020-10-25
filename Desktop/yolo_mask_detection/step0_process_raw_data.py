from bs4 import BeautifulSoup
import os
import shutil

image_path = os.path.join('.' , 'images')
label_path = os.path.join('.' , 'labels')
yolo_path = os.path.join('.' , 'yolo')
class_dic = {'none': 0 , 'bad': 1 , 'good': 2}

for idx , filename in enumerate(os.listdir(label_path)):

    try:
        f = open(os.path.join(label_path , filename))
        soup = BeautifulSoup(f.read() , 'xml')
        imgname = soup.select('filename')[0].text
        image_w = int(soup.select('width')[0].text)
        image_h = int(soup.select('height')[0].text)
    
        record = []
        for obj in soup.select('object'):
            xmin = int(obj.select('xmin')[0].text)
            xmax = int(obj.select('xmax')[0].text)
            ymin = int(obj.select('ymin')[0].text)
            ymax = int(obj.select('ymax')[0].text)
            obj_class = class_dic[obj.select('name')[0].text]
    
            x = (xmin + (xmax - xmin)/2) * 1. / image_w
            y = (ymin + (ymax - ymin)/2) * 1. / image_h
            w = (xmax-xmin) * 1. / image_w
            h = (ymax-ymin) * 1. / image_h
            record.append(' '.join([str(obj_class) , str(x) , str(y) , str(w) , str(h)]))
    
        newname = str(idx)
        if os.path.exists(os.path.join(image_path , imgname)):
            original_file = os.path.join(image_path , imgname)
            new_file = os.path.join(yolo_path , newname + '.jpg')
            shutil.copyfile(original_file , new_file)
            new_file = open(os.path.join(yolo_path , newname + '.txt') , 'w')
            new_file.write('\n'.join(record))

    except Exception as e:
        print(e)

