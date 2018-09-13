clc;clear;
fileID = fopen('rcv1_topics.txt','r');
data = textscan(fileID,'%s %f %*[^\n]');
fclose(fileID);
class = char(data{1});
class = class(:,1);
doc = data{2};
% 30000 samples 15000 training 15000 testing
doc = doc(1:95201);
class = class(1:95201);
% doc_id = unique(doc)-2285;
tongji = zeros(30000,4);
temp = doc(1);
j = 1;
for i = 1:95201
    if temp < doc(i)
        temp = doc(i);
        j=j+1;
    end
   switch class(i) 
       case 'E'
           tongji(j,1)=tongji(j,1)+1;
       case 'M'
           tongji(j,2)=tongji(j,2)+1;
       case 'C'
           tongji(j,3)=tongji(j,3)+1;
       case 'G'
           tongji(j,4)=tongji(j,4)+1;
   end
end
[~,label] = max(tongji,[],2);
labeltrn=label(1:21000);
labeltst=label(21001:end);