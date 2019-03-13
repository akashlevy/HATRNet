clear all;
close all;

fileName = 'hyperparam_results_coarse.csv';
results = csvread(fileName,1,0);

conv1_block = results(:,2);
base_filter_num = results(:,3);
conv1_kernel = results(:,4);
conv2_kernel = results(:,5);
dense_size = results(:,6);
drop = results(:,7);
batch_size = results(:,8);
learning_rate = results(:,9);
avg_loss_validation = results(:,10);
avg_accuracy_validation = results(:,11);

% Learning rate
[lr_sort,i] = sort(learning_rate);
% Filter
N = 80;
filter_coeff = ones(1,N)/N;
avg_accuracy = filtfilt(filter_coeff, 1, [avg_accuracy_validation(i(1:N));avg_accuracy_validation(i)]);
figure();
semilogx(lr_sort,avg_accuracy_validation(i)*100);
hold on;
semilogx(lr_sort,avg_accuracy(N+1:end)*100, 'LineWidth', 1.5);
xlabel('Learning rate');
ylabel('Validation accuracy in %');
legend('Raw data', 'Smoothed data');

% Delete runs with bad accuracy due to high learning rate
s=1;e=450;
conv1_block = conv1_block(i(s:e)); base_filter_num = base_filter_num(i(s:e));
conv1_kernel = conv1_kernel(i(s:e)); conv2_kernel = conv2_kernel(i(s:e));
dense_size = dense_size(i(s:e)); drop = drop(i(s:e));
batch_size = batch_size(i(s:e)); learning_rate = learning_rate(i(s:e));
avg_loss_validation = avg_loss_validation(i(s:e)); 
avg_accuracy_validation = avg_accuracy_validation(i(s:e));

% Base filter number
i=1; stop=max(base_filter_num);
while i<=stop
    k = find(base_filter_num==i);
    if ~isempty(k)
        bfn_median(i) = median(avg_accuracy_validation(k)*100);
    end
    i=i+1;
end
% Filter
N = 5;
filter_coeff = ones(1,N)/N;
bfn_median_filt = filtfilt(filter_coeff, 1, [bfn_median(1:N),bfn_median]);
figure();
stem(1:stop,bfn_median);
hold on;
plot(1:stop,bfn_median_filt(6:end), 'LineWidth', 1.5);
xlabel('Number of filters in first 1D convolution block');
ylabel('Median validation accuracy in %');

% Number of 1D Conv blocks
i=1; stop=max(conv1_block);
while i<=stop
    k = find(conv1_block==i);
    if ~isempty(k)
        c1block_median(i) = median(avg_accuracy_validation(k)*100);
    end
    i=i+1;
end
figure();
stem(1:stop,c1block_median);
xlabel('Number of 1D convolution blocks');
ylabel('Validation accuracy in %');

% Size of 1D Kernel
i=1; stop=max(conv1_kernel);
while i<=stop
    k = find(conv1_kernel==i);
    if ~isempty(k)
        c1kernel_median(i) = median(avg_accuracy_validation(k)*100);
    end
    i=i+1;
end
figure();
stem(1:stop,c1kernel_median);
xlabel('1D convolution kernel size nxn');
ylabel('Validation accuracy in %');

% Size of 2D Kernel
i=1; stop=max(conv2_kernel);
while i<=stop
    k = find(conv2_kernel==i);
    if ~isempty(k)
        c2kernel_median(i) = median(avg_accuracy_validation(k)*100);
    end
    i=i+1;
end
figure();
stem(1:stop,c2kernel_median);
xlabel('2D convolution kernel size nxn');
ylabel('Validation accuracy in %');

% Size of dense layer
i=1; stop=max(dense_size);
while i<=stop
    k = find(dense_size==i);
    if ~isempty(k)
        dsize_median(i) = median(avg_accuracy_validation(k)*100);
    end
    i=i+1;
end
figure();
stem(1:stop,dsize_median);
xlabel('Number of neurons in fully-connected layer');
ylabel('Validation accuracy in %');

% Dropout
[drop_sort,i] = sort(drop);
% Filter
N = 80;
filter_coeff = ones(1,N)/N;
avg_accuracy = filtfilt(filter_coeff, 1, [avg_accuracy_validation(i(1:N));avg_accuracy_validation(i)]);
figure();
plot(drop_sort*100,avg_accuracy_validation(i)*100);
hold on;
plot(drop_sort*100,avg_accuracy(N+1:end)*100, 'LineWidth', 1.5);
xlabel('Dropout in %');
ylabel('Validation accuracy in %');
legend('Raw data', 'Smoothed data');
