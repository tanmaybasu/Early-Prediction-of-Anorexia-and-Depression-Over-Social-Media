

def feature_generation_from_metamap_output(path,semantic_types,mm,rm,nfm,cl,opt): 
    note = codecs.open(path,encoding='utf-8',mode='r').readlines()
    rm=rm+1;
    mm[rm].insert(0,cl)                  # Inserting class label
    for l in range(0,len(note)): 
        line=note[l].strip('\n')
        #print line
        result=re.search( r'(.+?)(\b[7-9][0-9][0-9]|1000\b)([ ]+)([C])([0-9]+)(:)(.+)', line)  # Metamap confidence is greater than or equal to 700 
        #result=re.search( r'(.+?)(\b1000\b)([ ]+)([C])([0-9]+)(:)(.+)', line)                 # Metamap confidence is equal to 1000 only 
        if result:
            match=result.group(0)
            #print match                    
            st=re.search( r'(\[)([a-zA-Z, ]+)(\])', match)
            if st and semantic_types.count(st.group(2)):      # Checking the semantic type
                ptrn=re.search( r'([C])([0-9]+)(:)', match)   # Extracting CID
                if ptrn:
                    cid=ptrn.group(1)+ptrn.group(2)
                    #print cid+'\t'+st.group(2)                                                  
                    count=0;
                    for i in range(1,nfm):           
                        if mm[0][i]==cid:        # Headers are stored in first row (=0)
                            mm[rm].insert(i,1)
                            count=1
                    if count!=1 and opt!=1:      # opt=1 ensuring that no extra features are added from the test set
                        nfm=nfm+1
                        mm[0].insert(nfm,cid)
                        mm[rm].insert(nfm,1) 
    return mm,rm,nfm                         
   
# Store metamap features to CSV 
def metamap_features_to_csv(mm,row,len_trn,path_all,path_refine):    
    nmf=len(max(mm, key=len)) 
#    print 'Total Metamap Features: '+str(nmf)   
    temp=[['0' for y in range(nmf)] for x in range(row)] 
    for i in range(0,row):
        for j in range(0,len(mm[i])):
            temp[i][j]=mm[i][j]
            
    print ('No. of All Metamap Features: '+str(len(temp[1]))       )             
    fla = open(path_all, 'wb')
    wr = csv.writer(fla, delimiter=',',dialect='excel')    
    for rw in temp:
        wr.writerow(rw) 
    fla.close()   

    # Removing rare features        
    refined_data=[]
    for i in range(0,nmf):                                 
        elm=[];     
        for EM in temp[1:len(temp)]:            # Ignoring the first row (heading)   
            elm.append(EM[i])   
        if i==0:                                # Since the first column contains class labels 
            refined_data.append(elm)             
        else:
            refined_data.append(elm)             
#            mce=max(set(elm[0:len_trn]), key=elm.count)    # Finding most common element 
#            count=elm.count(mce)
#            if count<0.95*len_trn:
#                refined_data.append(elm)
    refined_data=map(list, zip(*refined_data))  # Transposing the list
    
    print ('No. of Refined Metamap Features: '+str(len(refined_data[1])-1)    ) # First column contains clas labels                  
    flr = open(path_refine, 'wb')
    wr = csv.writer(flr, delimiter=',',dialect='excel')    
    for rw in refined_data:
        wr.writerow(rw) 
    flr.close() 
    
def main(): 
    inp1 = raw_input("Choose : \n\t '1' to process raw text \n\t '2' to process metamap features from text \n\n")
    inp2 = raw_input("Choose : \n\t '1' to use the three annotators data for training \n\t '2' to use both one annotator and three annotators data for training \n\n")
#    semantic_types=['Individual Behavior','Mental Process','Mental or Behavioral Dysfunction'] 
               
    semantic_types=['Biomedical Occupation or Discipline',                           # Initially we selected this set 
                    'Diagnostic Procedure','Disease or Syndrome','Finding',
                    'Laboratory Procedure','Laboratory or Test Result',
                    'Mental Process','Mental or Behavioral Dysfunction',
                    'Organic Chemical','Pharmacologic Substance','Sign or Symptom',
                    'Social Behavior','Natural Phenomenon or Process', 'Behavior']              
     # Process metamap features in the 3-annotators training data               
     mmr=[[] for x in range(row)]
     mmr[0].insert(0,'class label')                # Writing the header Class Label in the top of first column 
     rmr=0; nrmf=1;         
     for i in range(0,len(fileread_3annotated_trn)):
     	fl = fileread_3annotated_trn[i].strip('\n') 
        path = st_3annotated_trn+fl
        mmr,rmr,nrmf=feature_generation_from_metamap_output(path,semantic_types,mmr,rmr,nrmf,cl_3annotated_trn[i],0) 
     metamap_features_to_csv(mmr,ln_trn+1,ln_trn,path_all_3annotated_trn,path_refine_3annotated_trn)
    
    
   
