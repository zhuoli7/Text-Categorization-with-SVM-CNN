import json
f=open('reviews_CDs_and_Vinyl_5.json','r')
data=[]
for line in f:
    data.append(json.loads(line))

write_file1 = 'data_10.txt'
output = open(write_file1,'w')
for i in range(100000):
    write_str1 = data[i]["reviewText"] + '\n'
    output.write(write_str1) 
output.close()

write_file2 = 'label_10.txt'
output = open(write_file2,'w')
for i in range(100000):
    write_str2 = str(data[i]["overall"]) + '\n'
    output.write(write_str2) 
output.close()
