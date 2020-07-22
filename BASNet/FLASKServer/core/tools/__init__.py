import base64, cv2
import torch
from torch.autograd import Variable
import numpy as np
from ..model import BASNet
from torchvision import transforms
from PIL import Image
from core.data_loader import RescaleT
from core.data_loader import ToTensorLab

class Detector(object):
    def __init__(self, model_dir):
        print ("...load BASNet...")
        self.net = BASNet (3, 1)
        self.net.load_state_dict (torch.load (model_dir))
        if torch.cuda.is_available ( ):
            self.net.cuda ( )
        self.net.eval ( )

    def normPRED(self,d):
        ma = torch.max (d)
        mi = torch.min (d)
        dn = (d - mi) / (ma - mi)
        return dn

    def image_to_base64(self, image_np: np.ndarray):
        image_cv2_encoded = cv2.imencode ('.png', image_np)[1]
        image_base64: str = str (base64.b64encode (image_cv2_encoded))[2:-1]
        return image_base64

    def base64_to_image(self,image_base64: str):
        image_cv2_encoded: str = base64.b64decode (image_base64)
        nparry: np.array = np.frombuffer (image_cv2_encoded, np.uint8)
        image_np: np.array = cv2.imdecode (nparry, cv2.IMREAD_COLOR)
        return image_np

    def detect(self, image_base64:str):
        try:
            image_np:np.array = self.base64_to_image(image_base64)
        except:
            print('ERROR2')

        print ("start inferencing")

        inputs_test = transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])(image_np)
        inputs_test = inputs_test.type (torch.FloatTensor)
        inputs_test = Variable (torch.unsqueeze (inputs_test, dim=0).float ( ), requires_grad=False)

        if torch.cuda.is_available ( ):
            inputs_test = Variable (inputs_test.cuda ( ))
        else:
            inputs_test = Variable (inputs_test)
        d1, d2, d3, d4, d5, d6, d7, d8 = self.net (inputs_test)
        # normalization
        pred = d1[:, 0, :, :]
        pred = self.normPRED (pred)
        del d1, d2, d3, d4, d5, d6, d7, d8
        predict = pred.squeeze ( )
        predict_np = predict.cpu ( ).data.numpy ()

        predict_pil = Image.fromarray (predict_np * 255).convert ('RGB')
        predict_np = np.asarray(predict_pil)

        predict_base64 = self.image_to_base64(predict_np)
        return predict_base64