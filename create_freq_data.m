clear all;
close all;

load('data/data1.mat'); data=data1;
Fs = 50;
T = 1/Fs;

data = data - mean(data,2);
data = data ./ std(data,0,2);

for i=1:size(data,1)
    [~,data_fft(i,:,1)] = fft_freq(data(i,:,1), Fs, 2000);
    [~,data_fft(i,:,2)] = fft_freq(data(i,:,2), Fs, 2000);
    [~,data_fft(i,:,3)] = fft_freq(data(i,:,3), Fs, 2000);
    [~,data_fft(i,:,4)] = fft_freq(data(i,:,4), Fs, 2000);
    [~,data_fft(i,:,5)] = fft_freq(data(i,:,5), Fs, 2000);
    [~,data_fft(i,:,6)] = fft_freq(data(i,:,6), Fs, 2000);
end
