%% read data
d1 = load('amazon_data_train_unig.mat');
d2 = load('amazon_data_test_unig.mat');
data_train = d1.mat;
data_test = d2.mat;

%% creat label
temp1 = zeros(12500,1);
temp2 = ones(12500,1);
label_train = [temp1;temp2];
label_test = [temp1;temp2];

%% training
% svm =
% svmtrain(mat_train,label_train,'kernel_function','linear','autoscale',false,'kernelcachelimit',16000000);
testrr=zeros(14,3);
Indices=crossvalind('Kfold',length(label_train),5);
for i=-3:1:10
    disp(i);
    box = 2.^i;
    for k = -2:1:0;
        sigma = 2.^k;
        testr=0;
        for j =1:5
            test_x=data_train(Indices~=j,:);
            train_x=data_train(Indices==j,:);
            test_y=label_train(Indices~=j);
            train_y=label_train(Indices==j);
            c1=box.*ones(length(train_y),1);
            tic;
            svm = svmtrain(train_x,train_y,'kernel_function','rbf','autoscale',false,'boxconstraint',c1,'rbf_sigma',sigma,'kernelcachelimit',10000000,'method','SMO');
            toc;
            resulta=svmclassify(svm,test_x);
            CM=confusionmat(resulta,test_y);
            diag_CM=diag(CM);
            testr=testr+sum(diag_CM)/length(test_y);
            disp(j);
        end
        disp(k);
        testrr(i+4,k+3)=testr/5;
    end
    disp(i);
end
% t=-5:1:15;
% plot(t,testrr);
% xlabel('logC');
% ylabel('test CCR');
% title('test CCR');
% disp('the best C* is:');
% [q,indexc]=max(testrr);
% disp(indexc);
t = -3:1:10;
sig = -2:1:0;

contourf(t,sig,testrr');
colorbar;
ylabel('Log(Sigma)');
xlabel('Log(c)');

ddd = find(testrr==max(max(testrr)));
[xx,yy] = ind2sub([14 3],ddd);
c_ind = t(xx);
sig_ind = sig(yy);

disp('start train/test');
c_star = 2.^(c_star-4);
c_star_vec = c_star.*ones(length(data_train),1);
sigma_star = 2.^(sig_ind-3);
svm = svmtrain(data_train,label_train,'kernel_function','rbf','autoscale',false,'boxconstraint',c_star_vec,'rbf_sigma',sigma_star,'kernelcachelimit',10000000,'method','SMO');
resulta = svmclassify(svm,data_test);
CM_test = confusionmat(resulta,label_test);
diag_CM = diag(CM_test);
cdc = sum(diag_CM)/length(label_test);
p = CM_test(2,2)/(sum(CM_test(:,2)));
r = CM_test(2,2)/(sum(CM_test(2,:),2));
f = 2*p*r/(p+r);