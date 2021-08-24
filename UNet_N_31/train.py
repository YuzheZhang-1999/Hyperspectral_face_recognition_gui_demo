import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage.measure as measure
from torch.utils.data import DataLoader
from model import UNet
import dataset_old as dataset
import xlwt
from math import log10


def main():
    dtype = 'float32'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Net define
    # Multiscale
    MSNet = UNet.MultiscaleNet().to(device)
    current_iter = 0
    MSNet.load_state_dict(torch.load('./checkpoint/checkpoint_9_9.pt'))
    best_psnr = 0
    cur_psnr = 0

    # training dataset#
    trainSet = dataset.MyDataSet('./data/UNet-hsi-train')
    # trainSet = dataset.MyDataSet('./data/UNet-hsi-train', './data/Scene')
    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    # validSet = dataset.MyDataSet('./data/UNet-hsi-valid-9')
    # validLoader = DataLoader(validSet, batch_size=6, shuffle=True, num_workers=1)
    # valid_freq = 2

    optimizer = optim.Adam(MSNet.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)
    criterion = nn.MSELoss()
    lossDirection = []


    for epoch in range(10, 100):
        running_loss = 0
        for i, data in enumerate(trainLoader, 0):
            current_iter += 1
            # if current_iter % valid_freq == 0:
            #     cur_psnr = valid(validLoader, MSNet, criterion, current_iter, device)

            MSNet.train()

            optimizer.zero_grad()
            trainmasaicPic, traingdPic, picName = data['masaic'].to(device),\
                                                  data['gd'].to(device),\
                                                  data['name'][0]
            output = MSNet(trainmasaicPic)
            loss = criterion(output, traingdPic)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            # if cur_psnr > best_psnr:
            #     best_psnr = max(cur_psnr, best_psnr)
            #     torch.save(MSNet.state_dict(), 'checkpoint/checkpoint_best.pt')


        print(str(epoch) + ': loss = ' + str(running_loss/len(trainLoader)))
        lossDirection.append(running_loss/len(trainLoader))
        if (epoch+1) % 10 == 0:
            torch.save(MSNet.state_dict(), 'checkpoint/checkpoint_9_' + str(epoch) + '.pt', _use_new_zipfile_serialization=False)

    outfile = xlwt.Workbook()
    sheet1 = outfile.add_sheet('sheet1', cell_overwrite_ok=False)
    for tempi in range(len(lossDirection)):
        sheet1.write(tempi, 0, lossDirection[tempi])
    lossPath = './loss/20210717-normalization' + '.xls'
    outfile.save(lossPath)

############################################################################################
#
#   functions
#
############################################################################################
def valid(validLoader, MSNet, criterion, iter, device):
    psnr = 0
    MSNet.eval()
    with torch.no_grad():
        for i, data in enumerate(validLoader):
            # valid
            trainmasaicPic, traingPic = data['mosaic'].to(device),\
                                        data['gd'].to(device)

            output = MSNet(trainmasaicPic)

            # psnr
            mse = criterion(output, traingPic).item()
            psnr += 10 * log10((1. ** 2) / mse)
            print('Iter: [{0}][{1}/{2}]\t''TEST PSNR: ({3})\t'.format(
                iter, i, len(validLoader), psnr/len(validLoader)))

    return psnr/len(validLoader)


if __name__ == '__main__':
    main()
