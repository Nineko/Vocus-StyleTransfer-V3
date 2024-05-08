import numpy as np
import torch
import os

from torch.autograd import Variable
from utils.imgprocess import img_transform_512,load_image,save_image
from net.transform import TransformNet

style_path = "ready2trans/content.jpg"
model_path = "models/MyStyle.pt"
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
style_id = 1

content = load_image(style_path)
content = img_transform_512(content)
content = content.unsqueeze(0)
content = Variable(content).to(device=device, dtype=dtype)

model = TransformNet().to(device=device, dtype=dtype)
model.load_state_dict(torch.load(model_path))

for a in range(4):
    stylized = model(content,a).cpu()
    save_image("results/result_"+str(a)+".jpg", stylized.data[0])
#stylized = model(content,style_id).cpu()
#save_image("results/result_"+str(style_id)+".jpg", stylized.data[0])