clc;clear;
load('ytest.mat');
load('ytrain.mat');
train_l=labeltrn;
test_l=labeltst;
% for i=1:21000
%     x=rand;
%     if x<1/3
%         train_l(i)=train_l(i)+4;
%     elseif x<2/3
%         train_l(i)=train_l(i)+8;
%     end
% end
% for i=1:9000
%     x=rand;
%     if x<1/3
%         test_l(i)=test_l(i)+4;
%     elseif x<2/3
%         test_l(i)=test_l(i)+8;
%     end
% end

for i=1:21000
    x=rand;
    if x<1/2
        train_l(i)=train_l(i)+4;
    end
end
for i=1:9000
    x=rand;
    if x<1/2
        test_l(i)=test_l(i)+4;
    end
end