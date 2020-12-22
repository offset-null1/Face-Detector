import numpy as np
import cv2
import imutils
import logging
import os,sys

fileName = sys.argv[0]

cwd = os.getcwd()

if fileName.startswith('.'):
    PATH = cwd + fileName[1:]
elif fileName.startswith('/'):
    PATH = fileName
else:
    PATH = cwd + '/' + fileName

logging.info(f' PATH to executable {PATH}')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    filename=PATH + '-application.log',
    format='%(asctime)s.%(msecs)-3d:%(filename)s:%(funcName)s:%(levelname)s:%(lineno)d:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)

class camera:
    def __init__(self,source=None):
        try:
            self.cap = cv2.VideoCapture(source)
            logging.debug(f'Camera obj:{self.cap}')
        except:
            logging.critical('Video capture failed, please check the given video source is valid')
            logging.critical('Returning...')
            return
            
        
    def __del__ (self):
        self.cap.release()
        logging.info(' VideoCapture source released')
        
    def getRawFrames(self):    
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            logging.warning(' VideoCapture source is not opened.')
            logging.critical(' Returning None')
            return


class detector(camera):
    def __init__(self,source=0):
        super(detector,self).__init__(source=source)
        
    def load_model(self,proto_path=None, weights_path=None):
        
        if proto_path is None:
            proto_path = '/home/daksh/opencv/samples/dnn/face_detector/deploy.prototxt'
        if weights_path is None:
            weights_path = '/home/daksh/opencv/samples/dnn/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        
        if os.path.isabs(proto_path) and os.path.isabs(weights_path):
            # logging.info(' Loading model..')
            net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
            return net
        else:
            logging.critical(f' Provide absolute paths, given paths: {proto_path}, {weights_path}')
            
    
    def detect(self,confidence=0.5):
        net = self.load_model()
        # logging.info(' Detecting...')
        frame = self.getRawFrames()
        frame = imutils.resize(frame,width=400)
        orig = frame.copy()
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
        
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            this_confidence = detections[0,0,i,2] 
            if this_confidence < confidence:
                continue
            box = detections[0,0,i,3:7]* np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')
            text = '{:.2f}%'.format(this_confidence*100)
            y = startY - 10 if startY-10>10 else startY+10
            cv2.rectangle(frame, (startX,startY),(endX,endY), (170,100,220),2) 
            cv2.putText(frame, text, (startX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(170,100,220), 2) 
            roi = orig[startY:endY,startX:endX]
            
        return orig,frame              
                

                
if __name__ == '__main__':
    print('Raw detector module')
else:
    pass