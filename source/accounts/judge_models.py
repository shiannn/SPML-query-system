import os
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from django.conf import settings

class JudgeModels(object):
    def __init__(self, model_root):
        self.resnet56_fgsm = ptcv_get_model("resnet56_cifar100")
        self.resnet56_fgsm.load_state_dict(
            torch.load(
                os.path.join(model_root, 'resnet56_cifar100_fgsm.pkl'),
                map_location=torch.device('cpu')
            )
        )
        self.resnet56_fgsm = self.resnet56_fgsm.eval()
        self.nin_fgsm = ptcv_get_model("nin_cifar100")
        self.nin_fgsm.load_state_dict(
            torch.load(
                os.path.join(model_root, 'nin_cifar100_fgsm.pkl'),
                map_location=torch.device('cpu')
            )
        )
        self.nin_fgsm = self.nin_fgsm.eval()
        self.resnet110_pgd = ptcv_get_model("resnet110_cifar100")
        self.resnet110_pgd.load_state_dict(
            torch.load(
                os.path.join(model_root, 'resnet110_cifar100_pgd.pkl'),
                map_location=torch.device('cpu')
            )
        )
        self.resnet110_pgd = self.resnet110_pgd.eval()
    def predict(self, img_tensor):
        ### get tensor with (1,3,32,32) return pred
        resnet56_fgsm_pred = self.resnet56_fgsm(img_tensor)
        resnet110_pgd_pred = self.resnet110_pgd(img_tensor)
        nin_fgsm_pred = self.nin_fgsm(img_tensor)
        resnet56_fgsm_pred = torch.nn.functional.softmax(resnet56_fgsm_pred, dim=1)
        resnet110_pgd_pred = torch.nn.functional.softmax(resnet110_pgd_pred, dim=1)
        nin_fgsm_pred = torch.nn.functional.softmax(nin_fgsm_pred, dim=1)
        pred = resnet56_fgsm_pred/3 + resnet110_pgd_pred/3 + nin_fgsm_pred/3
        
        return pred

#JudgeModels()
#judgeModels = JudgeModels()
ACCOUNT_DIR = os.path.join(settings.BASE_DIR, 'accounts')
judgeModels = JudgeModels(model_root=os.path.join(ACCOUNT_DIR, 'judgeModels'))