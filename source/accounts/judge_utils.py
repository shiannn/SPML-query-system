import os
import csv
import zipfile
import numpy as np
import cv2
from pathlib import Path
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
#from django.conf import settings

def handle_uploaded_images(user_dir, file_name):
    print('handle_uploaded_images')
    net = ptcv_get_model("resnet20_cifar100", pretrained=True)
    net = net.eval()
    print(user_dir, file_name)
    mean_rgb = torch.tensor([0.4914, 0.4822, 0.4465])
    std_rgb = torch.tensor([0.2023, 0.1994, 0.2010])
    acc = 0
    ### unzip
    header = ['name', 'logit1', 'logit2', 'logit3', 'logit4', 'logit5']
    datas = []
    with zipfile.ZipFile(os.path.join(user_dir, file_name)) as myzip:
        with torch.no_grad():
            for idx,file_ in enumerate(myzip.infolist()):
                print(file_.filename)
                if Path(file_.filename).suffix in ['.png', '.jpg']:
                    input_name = Path(file_.filename).name
                    tgt = int(input_name.split('_')[0])
                    single_file = myzip.read(file_)
                    img = cv2.imdecode(np.frombuffer(single_file, np.uint8), 1)
                    #if idx == 23 or idx == 87 or idx == 104 or idx == 61 or idx == 18:
                    #    cv2.imwrite(Path(file_.filename).name, img)
                    ### img (32,32,3)
                    img_tensor = torch.tensor(img).float() / 255.0
                    img_tensor = (img_tensor - mean_rgb) / std_rgb
                    img_tensor = img_tensor.permute(2,0,1).unsqueeze(0)
                    ### predict (1,3,32,32)
                    print('img_tensor.shape', img_tensor.shape)
                    ### output csv and save
                    pred = net(img_tensor)
                    top5val, top5idx = torch.topk(pred, k=5, dim=1)
                    #maxidx = pred.argmax()
                    print(top5idx[0,0].item(), 'tgt', tgt)
                    acc += (top5idx[0,0].item()==tgt)
                    data = [input_name]
                    data.extend(top5idx[0].tolist())
                    datas.append(data)
    print('acc', acc)
    print(datas)
    with open(os.path.join(user_dir, 'result.csv'), 'w') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for data in datas:
            writer.writerow(data)