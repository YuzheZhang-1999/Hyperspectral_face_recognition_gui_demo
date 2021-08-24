clc, 
clear, 
close all

%% Set parameter
datapath = './results';
resultpath = './checkpoint';
%%
fileread = dir(datapath);
psnr_allscene = [];
ssim_allscene = [];
for i = 3:size(fileread,1)
    load([datapath, '/', fileread(i).name]);
    gt = orig;
% PSNR of one scene
    for j = 1:size(gt,3)
        temp_psnr(j,1) = psnr(test(:,:,j), gt(:,:,j), max(max(gt(:,:,j))));
        temp_ssim(j,1) = ssim(test(:,:,j), gt(:,:,j));
    end
    psnr_allscene = [psnr_allscene, temp_psnr];
    ssim_allscene = [ssim_allscene, temp_ssim];
    fprintf('Idx of min psnr: %d\n', find(temp_psnr == min(temp_psnr)))
end

% % xlswrite([resultpath, '/Evaluation by Matlab.xls'], Evaluation)

    
    
