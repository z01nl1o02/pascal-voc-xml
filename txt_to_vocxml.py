import os
import sys
import json
import cv2
from pascal_voc_utils import PascalVocWriter


class VocXML:

    classes = []

    def __init__(self, classes_path):
        if not self.__load_classes_info(json_path=classes_path):
            sys.stderr.write("Error on loading the predefined classes info.\n")
            sys.exit(0)

    def __load_classes_info(self, json_path):
        with open(json_path,"rb") as f:  # for the not unicode string
            str_data = f.read()
            if str_data != '':
                self.classes = json.loads(str_data)
                return True
            return False

    def __load_object_info(self, txt_path, img_shape):
        H,W,C = img_shape
        objs = []
        with open(txt_path, "rb") as f:
            for line in f:
                splies = line.strip().split(" ")
                if len(splies) != 5:
                    sys.stderr.write("Error on reading the txt file{}.\n".format(txt_path))
                    sys.exit(0)
                    return False
                else:
                    cls,cx,cy,w,h = map(lambda x:float(x), splies)
                    x0 = cx - w/2
                    y0 = cy - h/2
                    x1 = x0 + w
                    y1 = y0 + h
                    x0,x1 = x0*W, x1*W
                    y0,y1 = y0*H, y1*H

                    cls = int(cls) + 1 #no background
                    points = [
                        (int(x0), int(y0)),
                        (int(x1), int(y1))
                    ]
                    objs.append({
                        "bndbox": self.__points2BndBox(points),
                        "label": self.classes[int(cls)]})
        return objs

    def __points2BndBox(self, points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        if xmin < 1:
            xmin = 1
        if ymin < 1:
            ymin = 1
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def save_to_vocxml(self, imagePath, bShow=True):
        imgDir, imgFileName = os.path.split(imagePath)
        imgFolderName = os.path.basename(imgDir)
        base = os.path.splitext(imagePath)[0]

        txt_path = base + ".txt"  # annotation file
        xml_path = base + ".xml"  # result file

        image = cv2.imread(imagePath,1)
        if image is not None and os.path.exists(txt_path):
            verified = True
            imageShape = image.shape[:]
        else:
            # verified = False
            return False

        # writer for pascalVoc format XML creater
        writer = PascalVocWriter(foldername=imgFolderName, filename=imgFileName, imgSize=imageShape,
                                 localImgPath=imagePath, verified=verified)

        # load the object info(border and class) from the txt file
        objs = self.__load_object_info(txt_path=txt_path, img_shape=image.shape)

        if bShow and verified:
            for obj in objs:
                [x1, y1, x2, y2] = obj["bndbox"]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.imshow("image", image)
            cv2.waitKey(0)

        for obj in objs:
            bndbox = obj["bndbox"]
            label = obj["label"]
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label)

        writer.save(targetFile=xml_path)
        return True

    def scan_dir(self, image_dir):
        img_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                     if os.path.splitext(fname)[1].lower() in [".png", ".jpg", ".jpeg"]]

        for img_path in img_paths:
            if self.save_to_vocxml(imagePath=img_path, bShow=False):
                print(img_path)


if __name__ == '__main__':
    vx = VocXML(classes_path="./data/classes.json")
    vx.scan_dir(image_dir='/home/data/traffic-objects/JPEGImages/')

