f = open('rcv1_qian30000.txt', 'r')
# f=open('test.txt','r')
data = [' ']
for line in f:
    if line[0] == '\n':
        data.append(' ')
    elif line[0] != '.':
        line = line[0:-2]
        data[-1] = data[-1] + ' ' + line
        # print(data[-1])
f2 = open('xdata.txt','w')
for i in range(21000):
    write_str1 = data[i][1:-1] + '\n'
    f2.write(write_str1) 
f2.close()

f3 = open('ydata.txt','w')
for i in range(9000):
    j = i + 21000
    write_str2 = data[j][1:-1] + '\n'
    f3.write(write_str2) 
f3.close()



