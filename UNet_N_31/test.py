import torch
import matplotlib.pyplot as plt
import skimage.measure as measure
from torch.utils.data import DataLoader
from model import MultiscaleNet_9
import dataset_unet as dataset
import scipy.io as sio
import numpy as np

dtype = 'float32'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Net define

# Multiscale
MSnet = MultiscaleNet_9.MultiscaleNet().to(device)
Name_checkpoint = 'iccv9_onlyiccv_novalid'
Idx_checkpoint = '99'
# PATH = './checkpoint/checkpoint_' + Name_checkpoint + '/checkpoint_best_' + Idx_checkpoint +'.pt'
PATH = './checkpoint' +  '/checkpoint_9_' + Idx_checkpoint +'.pt'
MSnet.load_state_dict(torch.load(PATH, map_location=device))
MSnet.eval()

# validation # '.\ExampleData\BGU' # or '.\ExampleData\camp'#
valSet = dataset.MyDataSet('./data/UNet-hsi-test-9-1')
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=0)
with torch.no_grad():
    for i, data in enumerate(valLoader, 0):
        valmasaicPic, valgdPic, picName = data['masaic'].to(device),\
                                          data['gd'].to(device),\
                                          data['name'][0]
        MSvaloutput = MSnet(valmasaicPic)
        # show the reconstruction of a certain channel
        channel = 1
        fig = plt.figure()
        gd = fig.add_subplot(141)
        gd.title.set_text('Ref')
        gdImg = valgdPic[0, channel, 0:, 0:].cpu().detach().numpy()
        gd.imshow(gdImg, cmap='gray')
        plt.axis('off')
        MSre = fig.add_subplot(143)
        MSreImg = MSvaloutput[0, channel, 0:, 0:].cpu().detach().numpy()
        MSpsnr = measure.compare_psnr(gdImg, MSreImg)
        MSre.title.set_text('MS\nPSNR: ' + str(round(MSpsnr, 2)))
        MSre.imshow(MSreImg, cmap='gray')
        plt.axis('off')
        plt.ioff()
        # plt.show()
        path = './results/results_' + Name_checkpoint + '_' + Idx_checkpoint + '/' + picName
        pred_np = MSvaloutput[0, :, :, :].cpu().detach().numpy()
        pred_np = np.transpose(pred_np, [1, 2, 0])
        gd_np = valgdPic[0, :, :, :].cpu().detach().numpy()
        gd_np = np.transpose(gd_np, [1, 2, 0])
        sio.savemat(path, {'test': pred_np, 'orig': gd_np})
