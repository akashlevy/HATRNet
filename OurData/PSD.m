clear all;
close all;

labels = importdata('labels.txt');
max_length = max(labels(:,5)-labels(:,4))+1;

for i=1:length(labels)
    acc = importdata(['acc_exp',num2str(labels(i,1),'%02.f'),'_user',num2str(labels(i,2),'%02.f'),'.txt']);
    gyro = importdata(['gyro_exp',num2str(labels(i,1),'%02.f'),'_user',num2str(labels(i,2),'%02.f'),'.txt']);
    
    data_start = labels(i,4);
    data_end = labels(i,5);
    
    % Data 1
    data{i} = [acc(data_start:data_end,:),gyro(data_start:data_end,:)];
    y(i,1) = labels(i,3);
    exp(i,1) = labels(i,1);
    user(i,1) = labels(i,2);
    
    % Data 2
    data1(i,:,:) = zeros(max_length,6);
    data1(i,1:data_end-data_start+1,:) = [acc(data_start:data_end,:),gyro(data_start:data_end,:)];
end