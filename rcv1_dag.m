clc;clear;
load('news_data_test.mat');
test=mat;
load('news_data_train.mat');
train=mat;
load('news_label8_test.mat');
load('news_label8_train.mat');
%train_l=labeltrn;
%test_l=labeltst;
result_s=zeros(9000,1);
for i=1:8
    disp(i);
    train_nx=train(train_l==i,:);
    train_ny=train_l(train_l==i);
    j=i+1;
    while j<9
        train_x=[train_nx;train(train_l==j,:)];
        train_y=[train_ny;train_l(train_l==j)];
        traina=svmtrain(sparse(train_x),train_y,'autoscale',false,'kernel_function','linear','kernelcachelimit',1000000);
        SVM_Model(i,j) = traina;
        j=j+1;
    end
end

% DAG
tic;
for k=1:9000
    % disp(k);
    test_c=test(k,:);
    i=1;
    j=8;
    for p=1:6
        r=svmclassify(SVM_Model(i,j),test_c);
        if r==i
            j=j-1;
        else
            i=i+1;
        end
    end
    r=svmclassify(SVM_Model(i,j),test_c);
    result_s(k,1)=r;
end
toc;
CM=confusionmat(result_s,test_l);
diag_CM=diag(CM);
testr=sum(diag_CM)/length(test_l);
disp('the overall CCR is:');
disp(testr);
disp('the confusion matrix of the test set:');
disp(CM);