import torch
import cv2
import model
import numpy as np
from PIL import Image
from torchvision import transforms
import os

class Enhancer:
    
    def __init__(self, frame_shape, scale_factor=12):
        self.w, self.h = frame_shape[0], frame_shape[1]
        self.model = model.enhance_net_nopool(scale_factor).cuda()
        self.model.load_state_dict(torch.load('weights/enhancer.pth'))
    
    def process(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = (np.asarray(image)/255.0)
        image = torch.from_numpy(image).float()
        image = image[0:self.h,0:self.w,:]
        image = image.permute(2,0,1)
        image = image.cuda().unsqueeze(0)
        enhanced_image, _ = self.model(image)
        enhanced_image = transforms.ToPILImage()(enhanced_image.squeeze_(0)).convert("RGB")
        return np.array(enhanced_image).astype('uint8')[:,:,::-1]
        

if __name__ == '__main__':
    input_vid_path = 'input/Test_0.avi'
    filename = os.path.basename(input_vid_path).split('.')[0]
    video = cv2.VideoCapture(input_vid_path)
    w = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//12)*12
    h = (int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//12)*12
    fps = int(video.get(cv2.CAP_PROP_FPS))
    enhancer = Enhancer(frame_shape=(w,h))
    writer = cv2.VideoWriter(os.path.join('output', filename+'.avi'), cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

    with torch.no_grad(): 
        while video.isOpened():
            ret, frame = video.read()
            
            if not ret:
                break
            out_image = enhancer.process(frame)
            writer.write(out_image)
     
    cv2.destroyAllWindows()
            
            
            
            
            
            
		

