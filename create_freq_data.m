clear all;
close all;

Fs = 50;

labels = importdata('data/HAPT_dataset/RawData/labels.txt');
max_length = max(labels(:,5)-labels(:,4))+1;

for i=1:length(labels)
    acc = importdata(['data/HAPT_dataset/RawData/acc_exp',num2str(labels(i,1),'%02.f'),'_user',num2str(labels(i,2),'%02.f'),'.txt']);
    gyro = importdata(['data/HAPT_dataset/RawData/gyro_exp',num2str(labels(i,1),'%02.f'),'_user',num2str(labels(i,2),'%02.f'),'.txt']);
    
    data_start = labels(i,4);
    data_end = labels(i,5);
    
    % Remove mean, normalize variance
    acc_data = acc(data_start:data_end,:);
    acc_data = acc_data - mean(acc_data);
    acc_data = acc_data ./ std(acc_data);
    gyro_data = gyro(data_start:data_end,:);
    gyro_data = gyro_data - mean(gyro_data);
    gyro_data = gyro_data ./ std(gyro_data);
    
    [~,data_fft(i,:,1)] = fft_freq(acc_data(:,1), Fs, 2000);
    [~,data_fft(i,:,2)] = fft_freq(acc_data(:,2), Fs, 2000);
    [~,data_fft(i,:,3)] = fft_freq(acc_data(:,3), Fs, 2000);
    [~,data_fft(i,:,4)] = fft_freq(gyro_data(:,1), Fs, 2000);
    [~,data_fft(i,:,5)] = fft_freq(gyro_data(:,2), Fs, 2000);
    [~,data_fft(i,:,6)] = fft_freq(gyro_data(:,3), Fs, 2000);    
end
