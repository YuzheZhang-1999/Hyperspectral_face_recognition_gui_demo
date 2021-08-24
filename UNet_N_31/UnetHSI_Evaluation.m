clc, 
clear, 
close all

%% Set parameter
datapath = './results/results_iccv9_onlyiccv_novalid_99';%修改31行对应处文件名
resultpath = './checkpoint/checkpoint_iccv9_onlyiccv_novalid';
%%
fileread = dir(datapath);
Evaluation(1,1) = {'Matname'};
Evaluation(1,2) = {'Average PSNR'};
Evaluation(1,3) = {'Average SSIM'};

for i = 3:size(fileread,1)
    load([datapath, '/', fileread(i).name]);
    gt = double(orig);
    test = double(test);
% PSNR of one scene
    for j = 1:size(gt,3)
        psnr_onescene(j,1) = psnr(test(:,:,j), gt(:,:,j), max(max(gt(:,:,j))));
        ssim_onescene(j,1) = ssim(test(:,:,j), gt(:,:,j));
    end
    fprintf('Idx of min psnr: %d\n', find(psnr_onescene == min(psnr_onescene)))
    mean_psnr = mean(psnr_onescene);
    mean_ssim = mean(ssim_onescene);
% Record all
    Evaluation(i-1,1) = {fileread(i).name};
    Evaluation(i-1,2) = {mean_psnr};
    Evaluation(i-1,3) = {mean_ssim};
end

xlswrite([resultpath, '/99-Evaluation by Matlab-double.xls'], Evaluation)

%% 看图
% % for i = 1:31
% %     diff = abs(orig(:,:,i)-test(:,:,i));
% %     figure(1)
% %     subplot(121)
% %     imshow(orig(:,:,i))
% %     subplot(122)
% %     imshow(test(:,:,i))
% %     figure(2)
% %     imshow(diff), colormap(jet), caxis([0,0.5]);
% %     title(num2str(i))
% %     waitforbuttonpress
% % end
    
    
