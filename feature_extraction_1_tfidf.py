from sklearn.feature_extraction.text import TfidfVectorizer 
import glob
import scipy.io as sio  
import numpy

voca = open('vocabulary_1.txt')
vocabu = list(voca.read().split())
# print(vocabu)

vectorizer = TfidfVectorizer(input='filename', min_df=1, vocabulary=vocabu )
train_list = []
neg_set = glob.glob('train\\neg\\*.txt')
# print(neg_set)
for i in neg_set:
	train_list.append(i)

pos_set = glob.glob('train\\pos\\*.txt')
for i in pos_set:
	train_list.append(i)


test_list = []
neg_test = glob.glob('test\\neg\\*.txt')
# print(neg_set)
for i in neg_test:
	test_list.append(i)

pos_test = glob.glob('test\\pos\\*.txt')
for i in pos_test:
	test_list.append(i)
# print(file_list)
train = vectorizer.fit_transform(train_list)
test = vectorizer.transform(test_list)
sio.savemat('train_tfidf.mat', {'mat':train})   
sio.savemat('test_tfidf.mat', {'mat':test})  
