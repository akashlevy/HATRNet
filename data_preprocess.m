clear all;
close all;

%% Load raw data (saved in .txt files)
addpath('data/HAPT_dataset/RawData')
labels = importdata('labels.txt');
max_length = max(labels(:,5)-labels(:,4))+1;


%% Loop through raw data .txt files and save them in different forms
for i=1:length(labels)
    % Load one piece of raw data
    acc = importdata(['acc_exp',num2str(labels(i,1),'%02.f'),'_user',num2str(labels(i,2),'%02.f'),'.txt']);
    gyro = importdata(['gyro_exp',num2str(labels(i,1),'%02.f'),'_user',num2str(labels(i,2),'%02.f'),'.txt']);
    data_start = labels(i,4);
    data_end = labels(i,5);
    
    % Raw data in cell array (variable lengths)
    data_raw{i} = [acc(data_start:data_end,:),gyro(data_start:data_end,:)];
    y_raw(i,1) = labels(i,3);
end
save('data/data_raw.mat','data_raw','y_raw');


%% Save timesliced data (no data augmentation)
% Slice data in n second pieces
Fs = 50; % Sampling rate
l = 2.5*Fs; % 2 seconds * sampling rate
data_timeslice = [];
y_timeslice = [];
for i=1:length(data_raw)
    data_start = 1;
    while data_start+l-1 < length(data_raw{i})
        data_timeslice(end+1,:,:) = data_raw{i}(data_start:data_start+l-1,:);
        data_start = data_start + l; 
        y_timeslice(end+1,1) = labels(i,3);
    end
end

% Train, Dev, Test Ratio
p_train = 0.6;
p_dev = 0.2;
p_test = 0.2;

% Split data
n = length(y_timeslice);
r = randperm(n);
for i=1:n
    if i <= floor(p_train*n)
        X_train(i,:,:) = data_timeslice(r(i),:,:);
        Y_train(i,1) = y_timeslice(r(i));
    elseif i <= floor((p_train+p_dev)*n)
        X_dev(i-floor(p_train*n),:,:) = data_timeslice(r(i),:,:);
        Y_dev(i-floor(p_train*n),1) = y_timeslice(r(i));
    else
        X_test(i-floor((p_train+p_dev)*n),:,:) = data_timeslice(r(i),:,:);
        Y_test(i-floor((p_train+p_dev)*n),1) = y_timeslice(r(i));
    end
end

% Zero mean, unit variance
X_train = X_train - mean(X_train,2);
X_train = X_train ./ std(X_train,0,2);
X_dev = X_dev - mean(X_dev,2);
X_dev = X_dev ./ std(X_dev,0,2);
X_test = X_test - mean(X_test,2);
X_test = X_test ./ std(X_test,0,2);

% Save
save('data/data_timeslice.mat','X_train','Y_train','X_dev','Y_dev','X_test','Y_test');
clear X_train Y_train X_dev Y_dev X_test Y_test;


%% Split raw data in train, dev and test and augment training data
% Train, Dev, Test Ratio
p_train = 0.6; % will get enhance by augmentation
p_dev = 0.2;
p_test = 0.2;

% Split data
n = length(data_raw);
r = randperm(n);
for i=1:n
    if i <= floor(p_train*n)
        X_train_raw{i} = data_raw{r(i)};
        Y_train_raw(i,1) = y_raw(r(i));
    elseif i <= floor((p_train+p_dev)*n)
        X_dev_raw{i-floor(p_train*n)} = data_raw{r(i)};
        Y_dev_raw(i-floor(p_train*n),1) = y_raw(r(i));
    else
        X_test_raw{i-floor((p_train+p_dev)*n)} = data_raw{r(i)};
        Y_test_raw(i-floor((p_train+p_dev)*n),1) = y_raw(r(i));
    end
end

% Augment each training data trace 4 times by squeezing stretching them
% with four different random factors
X_train_aug = {};
Y_train_aug = [];
for i=1:length(X_train_raw)
    X_train_aug{end+1} = X_train_raw{i};
    Y_train_aug(end+1,1) = Y_train_raw(i);
    for j=1:4 % Four augmentations per training example
        r = 0.75 + 0.5*rand(1); % Factor between 0.75 and 1.25
        x_old = linspace(0,1,length(X_train_raw{i}));
        x_new = linspace(0,1,round(r*length(X_train_raw{i})));
        temp1 = interp1(x_old,X_train_raw{i}(:,1),x_new);
        temp2 = interp1(x_old,X_train_raw{i}(:,2),x_new);
        temp3 = interp1(x_old,X_train_raw{i}(:,3),x_new);
        temp4 = interp1(x_old,X_train_raw{i}(:,4),x_new);
        temp5 = interp1(x_old,X_train_raw{i}(:,5),x_new);
        temp6 = interp1(x_old,X_train_raw{i}(:,6),x_new);
        X_train_aug{end+1} = [temp1',temp2',temp3',temp4',temp5',temp6'];
        Y_train_aug(end+1,1) = Y_train_raw(i);
    end
end


%% Save zero padded data
% Find longest array (could have changed due to data augmentation)
L = max(cellfun(@length,[X_train_aug,X_dev_raw,X_test_raw]));

for i=1:length(X_train_aug)
    X_train(i,:,:) = [X_train_aug{i}; zeros(L-length(X_train_aug{i}),6)];
    X_train_dup(i,:,:) = [X_train_aug{i}; X_train_aug{i}; zeros(2*L-2*length(X_train_aug{i}),6)];
end
Y_train = Y_train_aug;
for i=1:length(X_dev_raw)
    X_dev(i,:,:) = [X_dev_raw{i}; zeros(L-length(X_dev_raw{i}),6)];
    X_dev_dup(i,:,:) = [X_dev_raw{i}; X_dev_raw{i}; zeros(2*L-2*length(X_dev_raw{i}),6)];
end
Y_dev = Y_dev_raw;
for i=1:length(X_test_raw)
    X_test(i,:,:) = [X_test_raw{i}; zeros(L-length(X_test_raw{i}),6)];
    X_test_dup(i,:,:) = [X_test_raw{i}; X_test_raw{i}; zeros(2*L-2*length(X_test_raw{i}),6)];
end
Y_test = Y_test_raw;

% Zero mean, unit variance
X_train = X_train - mean(X_train,2);
X_train = X_train ./ std(X_train,0,2);
X_dev = X_dev - mean(X_dev,2);
X_dev = X_dev ./ std(X_dev,0,2);
X_test = X_test - mean(X_test,2);
X_test = X_test ./ std(X_test,0,2);

% Save
save('data/data_zeropad.mat','X_train','Y_train','X_dev','Y_dev','X_test','Y_test');


%% Save zero padded data concatenated with frequency data
% Perform FFT on raw data (including augmented data)
Fs = 50;
T = 1/Fs;
L = size(X_train,2); % length of longest time-series

% FFT Training data
for i=1:length(X_train_aug)
    [~,X_train_fft(i,:,1),X_train_fft(i,:,4)] = fft_freq(X_train_aug{i}(:,1), Fs, 2*L);
    [~,X_train_fft(i,:,2),X_train_fft(i,:,5)] = fft_freq(X_train_aug{i}(:,2), Fs, 2*L);
    [~,X_train_fft(i,:,3),X_train_fft(i,:,6)] = fft_freq(X_train_aug{i}(:,3), Fs, 2*L);
    [~,X_train_fft(i,:,7),X_train_fft(i,:,10)] = fft_freq(X_train_aug{i}(:,4), Fs, 2*L);
    [~,X_train_fft(i,:,8),X_train_fft(i,:,11)] = fft_freq(X_train_aug{i}(:,5), Fs, 2*L);
    [~,X_train_fft(i,:,9),X_train_fft(i,:,12)] = fft_freq(X_train_aug{i}(:,6), Fs, 2*L);
end
X_train_fft = X_train_fft(:,1:end-1,:); % cut last sample (fft generates L+1 samples)

% FFT Dev data
for i=1:length(X_dev_raw)
    [~,X_dev_fft(i,:,1),X_dev_fft(i,:,4)] = fft_freq(X_dev_raw{i}(:,1), Fs, 2*L);
    [~,X_dev_fft(i,:,2),X_dev_fft(i,:,5)] = fft_freq(X_dev_raw{i}(:,2), Fs, 2*L);
    [~,X_dev_fft(i,:,3),X_dev_fft(i,:,6)] = fft_freq(X_dev_raw{i}(:,3), Fs, 2*L);
    [~,X_dev_fft(i,:,7),X_dev_fft(i,:,10)] = fft_freq(X_dev_raw{i}(:,4), Fs, 2*L);
    [~,X_dev_fft(i,:,8),X_dev_fft(i,:,11)] = fft_freq(X_dev_raw{i}(:,5), Fs, 2*L);
    [~,X_dev_fft(i,:,9),X_dev_fft(i,:,12)] = fft_freq(X_dev_raw{i}(:,6), Fs, 2*L);
end
X_dev_fft = X_dev_fft(:,1:end-1,:); % cut last sample (fft generates L+1 samples)

% FFT Test data
for i=1:length(X_test_raw)
    [~,X_test_fft(i,:,1),X_test_fft(i,:,4)] = fft_freq(X_test_raw{i}(:,1), Fs, 2*L);
    [~,X_test_fft(i,:,2),X_test_fft(i,:,5)] = fft_freq(X_test_raw{i}(:,2), Fs, 2*L);
    [~,X_test_fft(i,:,3),X_test_fft(i,:,6)] = fft_freq(X_test_raw{i}(:,3), Fs, 2*L);
    [~,X_test_fft(i,:,7),X_test_fft(i,:,10)] = fft_freq(X_test_raw{i}(:,4), Fs, 2*L);
    [~,X_test_fft(i,:,8),X_test_fft(i,:,11)] = fft_freq(X_test_raw{i}(:,5), Fs, 2*L);
    [~,X_test_fft(i,:,9),X_test_fft(i,:,12)] = fft_freq(X_test_raw{i}(:,6), Fs, 2*L);
end
X_test_fft = X_test_fft(:,1:end-1,:); % cut last sample (fft generates L+1 samples)


% Concatenate with time data
X_train = cat(3,X_train,X_train_fft);
X_dev = cat(3,X_dev,X_dev_fft);
X_test = cat(3,X_test,X_test_fft);

% Save concatenated data
save('data/data_time_and_fft.mat','X_train','Y_train','X_dev','Y_dev','X_test','Y_test');

% Save frequency only
X_train = X_train_fft;
X_dev = X_dev_fft;
X_test = X_test_fft;
save('data/data_fft.mat','X_train','Y_train','X_dev','Y_dev','X_test','Y_test');


%% Save duplicated + zeropadded data concatenated with frequency data
clear X_train_fft X_dev_fft X_test_fft;
% Perform FFT on raw data (including augmented data)
Fs = 50;
T = 1/Fs;
L = size(X_train_dup,2); % length of longest time-series

% FFT Training data
for i=1:length(X_train_aug)
    X_temp_dup = [X_train_aug{i};X_train_aug{i}];
    [~,X_train_fft(i,:,1),X_train_fft(i,:,4)] = fft_freq(X_temp_dup(:,1), Fs, 2*L);
    [~,X_train_fft(i,:,2),X_train_fft(i,:,5)] = fft_freq(X_temp_dup(:,2), Fs, 2*L);
    [~,X_train_fft(i,:,3),X_train_fft(i,:,6)] = fft_freq(X_temp_dup(:,3), Fs, 2*L);
    [~,X_train_fft(i,:,7),X_train_fft(i,:,10)] = fft_freq(X_temp_dup(:,4), Fs, 2*L);
    [~,X_train_fft(i,:,8),X_train_fft(i,:,11)] = fft_freq(X_temp_dup(:,5), Fs, 2*L);
    [~,X_train_fft(i,:,9),X_train_fft(i,:,12)] = fft_freq(X_temp_dup(:,6), Fs, 2*L);
end
X_train_fft = X_train_fft(:,1:end-1,:); % cut last sample (fft generates L+1 samples)

% FFT Dev data
for i=1:length(X_dev_raw)
    X_temp_dup = [X_dev_raw{i};X_dev_raw{i}];
    [~,X_dev_fft(i,:,1),X_dev_fft(i,:,4)] = fft_freq(X_temp_dup(:,1), Fs, 2*L);
    [~,X_dev_fft(i,:,2),X_dev_fft(i,:,5)] = fft_freq(X_temp_dup(:,2), Fs, 2*L);
    [~,X_dev_fft(i,:,3),X_dev_fft(i,:,6)] = fft_freq(X_temp_dup(:,3), Fs, 2*L);
    [~,X_dev_fft(i,:,7),X_dev_fft(i,:,10)] = fft_freq(X_temp_dup(:,4), Fs, 2*L);
    [~,X_dev_fft(i,:,8),X_dev_fft(i,:,11)] = fft_freq(X_temp_dup(:,5), Fs, 2*L);
    [~,X_dev_fft(i,:,9),X_dev_fft(i,:,12)] = fft_freq(X_temp_dup(:,6), Fs, 2*L);
end
X_dev_fft = X_dev_fft(:,1:end-1,:); % cut last sample (fft generates L+1 samples)

% FFT Test data
for i=1:length(X_test_raw)
    X_temp_dup = [X_test_raw{i};X_test_raw{i}];
    [~,X_test_fft(i,:,1),X_test_fft(i,:,4)] = fft_freq(X_temp_dup(:,1), Fs, 2*L);
    [~,X_test_fft(i,:,2),X_test_fft(i,:,5)] = fft_freq(X_temp_dup(:,2), Fs, 2*L);
    [~,X_test_fft(i,:,3),X_test_fft(i,:,6)] = fft_freq(X_temp_dup(:,3), Fs, 2*L);
    [~,X_test_fft(i,:,7),X_test_fft(i,:,10)] = fft_freq(X_temp_dup(:,4), Fs, 2*L);
    [~,X_test_fft(i,:,8),X_test_fft(i,:,11)] = fft_freq(X_temp_dup(:,5), Fs, 2*L);
    [~,X_test_fft(i,:,9),X_test_fft(i,:,12)] = fft_freq(X_temp_dup(:,6), Fs, 2*L);
end
X_test_fft = X_test_fft(:,1:end-1,:); % cut last sample (fft generates L+1 samples)

% Concatenate with time data
X_train = cat(3,X_train_dup,X_train_fft);
X_dev = cat(3,X_dev_dup,X_dev_fft);
X_test = cat(3,X_test_dup,X_test_fft);

% Save concatenated data
save('data/data_time_and_fft_dup.mat','X_train','Y_train','X_dev','Y_dev','X_test','Y_test','-v7.3');

