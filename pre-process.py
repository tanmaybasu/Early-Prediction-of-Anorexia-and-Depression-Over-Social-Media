# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:20:29 2018

@author: Kalyani Jandhyala
@email : sree.kalyani95@gmail.com
"""
import os
import numpy as np
import pandas as pd
import pickle
from io import StringIO
import xml.etree.ElementTree as ET

prefix='/home/jandhyala/Sri/SEMISTER 4'

corpus =[]
### MAIN PROGRAM
################################# training: ############################
### FIRST STEP. READ XMLs and concatenate all text written by the subjects  chunkwise:

pos_folder='%s/pos_chunks 2' % prefix
neg_folder='%s/neg_chunks 2' % prefix

listfiles=os.listdir(pos_folder)
listfiles2=os.listdir(neg_folder)

# stores the number of writings for each user
cor1 = []  
cor1a =[]  
cor1b =[]   
for file in listfiles:
  filepath=os.path.join(pos_folder,file)  #print(filepath)
  tree = ET.parse(filepath)
  root = tree.getroot() #prints ID
  #print(root[0].text)
  pos_txt='' 
  for child in root:
    if child.tag!='ID':
        pos_txt= pos_txt+child[0].text+child[3].text 
  cor1.append(root[0].text+'\n'+pos_txt)
  cor1a.append(pos_txt)
  cor1b.append(root[0].text)
depPosWithClassLabels = []
for i in cor1a:
    row = []
    row.append(i)
    row.append(1)
    depPosWithClassLabels.append(row)
#cor1b = pd.DataFrame(cor1b)
# if child is not ID then it has to be a writing element
# child 0 is title, 1 is date, 2 is info and 3 is text # print(child[0].text)
cor2a = []
cor2 =[]
corb = []
for file in listfiles2:
  filepath=os.path.join(neg_folder,file)
  #print(filepath)
  try:
    tree = ET.parse(filepath)
  except Exception:
    continue
  root = tree.getroot()
  neg_txt='' 
  for child in root:
    if child.tag!='ID':
       neg_txt= neg_txt+child[0].text+child[3].text
  cor2.append(root[0].text+'\n'+neg_txt) #prints subject id with text
  cor2a.append(neg_txt)
  corb.append(root[0].text)
#corb = pd.DataFrame(corb)
depNegWithClassLabels = []
for i in cor2a:
    row = []
    row.append(i)
    row.append(0)
    depNegWithClassLabels.append(row)

corpus = cor1+cor2
type(corpus)

depression = depPosWithClassLabels + depNegWithClassLabels
depression = pd.DataFrame(depression,columns=['Text','Label'])
#depression1 = pd.DataFrame(depression['Label'])
depre = cor1b+corb
depre = pd.DataFrame(depre,columns = ['Id'])
#for aggreate trains
Depression_train  = [depression,depre]
Depression_train = pd.concat(Depression_train , axis=1)

Detrain = Depression_train.groupby('Id', sort=False)['Text'].apply(' '.join)
#Detrain.to_csv("Dep_train.csv" , sep = ',',index = True)
Depression_train1 = pd.read_csv("/home/jandhyala/Depression_CLEF/Dep_train.csv",header=None)  ##train data
Depression_train1['Label'] = np.zeros([len(Depression_train1)])
Depression_train1['Label'][0:83]=1
Depression_train1.columns=['Id','Text','Label']
Depression_train1.to_csv("agg_train.csv",sep =',',index=False)
###################################################
##setting target train:
#y_train = np.zeros([len(cor1)+len(cor2)])

#    y_train = np.zeros(len(corpus))
#    y_train[0:830]=1 #i.e, files bellonging to pos clss are assigned 1
###
#y_train = pd.DataFrame(y_train)
#y_train.columns = ['class_label']


#############################3

################ READ XMLs and concatenate all text written by the subjects  chunkwise:#############
test = '%s/all_test_2018' % prefix
tstfiles=os.listdir(test)

tst1 =[] 
tst1a =[]   
tst1b = []     
for file in tstfiles:
  filepath1=os.path.join(test,file)  #print(filepath)
  tree = ET.parse(filepath1)
  root = tree.getroot()
  root[0].text#prints ID
  #print(root[0].text)
  all_txt='' 
  for child in root:
    if child.tag!='ID':
         all_txt=all_txt+child[0].text+child[3].text
  tst1a.append(root[0].text+'\n'+all_txt)  #prints subject id with text
  tst1.append(all_txt)
  tst1b.append(root[0].text)   ##only id"s
tst1b=pd.DataFrame(tst1b)
tst1=pd.DataFrame(tst1)
##############################################################
#y_test =  np.zeros([len(tst1b)])
#y_test=pd.DataFrame(y_test)
#y_test[0:1000]=1

######### wrting xml txt to .txt file ########
def idf_text(doc):
    subject_id,text = doc.split('\n', 1)    
    return subject_id[-17:],text.strip()
split_data = sorted(map(idf_text,tst1a))   ###### corpus for train,tst1a for test.
split_data = pd.DataFrame(split_data,columns = ['ID','Text'])
type(split_data)        
data = split_data.groupby('ID',sort=False).agg('sum')
data.to_csv("agg_test_new.csv" , sep = ',',index = True)

