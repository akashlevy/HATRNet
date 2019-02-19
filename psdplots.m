clear all;
close all;

load('data1.mat'); data=data1;
load('y.mat')
load('length.mat');
Fs = 50;
T = 1/Fs;

% Find one walking series
walking = data(find(y==1),:,:);
len_walk = len(find(y==1));
walking = squeeze(walking(1,1:500,:));

% Find one sitting series
sitting = data(find(y==4),:,:);
len_sit = len(find(y==4));
sitting = squeeze(sitting(2,101:600,:));

% Find one standing series
standing = data(find(y==5),:,:);
len_sta = len(find(y==5));
standing = squeeze(standing(19,1:500,:));

% Remove mean, normalize variance
walking = walking - mean(walking);
walking = walking ./ std(walking);
sitting = sitting - mean(sitting);
sitting = sitting ./ std(sitting);
standing = standing - mean(standing);
standing = standing ./ std(standing);


%% Plots
% Time data walking
t = linspace(0,T*(length(walking)-1),length(walking));
acc = walking(:,1:3);
gyro = walking(:,4:6);
figure
subplot(2,1,1)
plot(t,acc,'linewidth',1)
subplot(2,1,2)
plot(t,gyro,'linewidth',1)

% Frequency data walking
[fx_acc_walk,yx_acc_walk] = fft_freq(acc(:,1), Fs);
[fy_acc_walk,yy_acc_walk] = fft_freq(acc(:,2), Fs);
[fz_acc_walk,yz_acc_walk] = fft_freq(acc(:,3), Fs);
[fx_gyro_walk,yx_gyro_walk] = fft_freq(gyro(:,1), Fs);
[fy_gyro_walk,yy_gyro_walk] = fft_freq(gyro(:,2), Fs);
[fz_gyro_walk,yz_gyro_walk] = fft_freq(gyro(:,3), Fs);
figure
subplot(2,1,1)
hold on
plot(fx_acc_walk,yx_acc_walk,'linewidth',1)
plot(fy_acc_walk,yy_acc_walk,'linewidth',1)
plot(fz_acc_walk,yz_acc_walk,'linewidth',1)
subplot(2,1,2)
hold on
plot(fx_gyro_walk,yx_gyro_walk,'linewidth',1)
plot(fy_gyro_walk,yy_gyro_walk,'linewidth',1)
plot(fz_gyro_walk,yz_gyro_walk,'linewidth',1) 

% Time data sitting
t = linspace(0,T*(length(sitting)-1),length(sitting));
acc = sitting(:,1:3);
gyro = sitting(:,4:6);
figure
subplot(2,1,1)
plot(t,acc,'linewidth',1)
subplot(2,1,2)
plot(t,gyro,'linewidth',1)

% Frequency data sitting
[fx_acc_sit,yx_acc_sit] = fft_freq(acc(:,1), Fs);
[fy_acc_sit,yy_acc_sit] = fft_freq(acc(:,2), Fs);
[fz_acc_sit,yz_acc_sit] = fft_freq(acc(:,3), Fs);
[fx_gyro_sit,yx_gyro_sit] = fft_freq(gyro(:,1), Fs);
[fy_gyro_sit,yy_gyro_sit] = fft_freq(gyro(:,2), Fs);
[fz_gyro_sit,yz_gyro_sit] = fft_freq(gyro(:,3), Fs);
figure
subplot(2,1,1)
hold on
plot(fx_acc_sit,yx_acc_sit,'linewidth',1)
plot(fy_acc_sit,yy_acc_sit,'linewidth',1)
plot(fz_acc_sit,yz_acc_sit,'linewidth',1)
subplot(2,1,2)
hold on
plot(fx_gyro_sit,yx_gyro_sit,'linewidth',1)
plot(fy_gyro_sit,yy_gyro_sit,'linewidth',1)
plot(fz_gyro_sit,yz_gyro_sit,'linewidth',1) 


% Time data sitting
t = linspace(0,T*(length(standing)-1),length(standing));
acc = standing(:,1:3);
gyro = standing(:,4:6);
figure
subplot(2,1,1)
plot(t,acc,'linewidth',1)
subplot(2,1,2)
plot(t,gyro,'linewidth',1)

% Frequency data sitting
[fx_acc_stand,yx_acc_stand] = fft_freq(acc(:,1), Fs);
[fy_acc_stand,yy_acc_stand] = fft_freq(acc(:,2), Fs);
[fz_acc_stand,yz_acc_stand] = fft_freq(acc(:,3), Fs);
[fx_gyro_stand,yx_gyro_stand] = fft_freq(gyro(:,1), Fs);
[fy_gyro_stand,yy_gyro_stand] = fft_freq(gyro(:,2), Fs);
[fz_gyro_stand,yz_gyro_stand] = fft_freq(gyro(:,3), Fs);
figure
subplot(2,1,1)
hold on
plot(fx_acc_stand,yx_acc_stand,'linewidth',1)
plot(fy_acc_stand,yy_acc_stand,'linewidth',1)
plot(fz_acc_stand,yz_acc_stand,'linewidth',1)
subplot(2,1,2)
hold on
plot(fx_gyro_stand,yx_gyro_stand,'linewidth',1)
plot(fy_gyro_stand,yy_gyro_stand,'linewidth',1)
plot(fz_gyro_stand,yz_gyro_stand,'linewidth',1) 
