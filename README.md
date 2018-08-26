# Early Detection of Signs of Anorexia and Depression Over Social Media using Effective Machine Learning Frameworks :

The CLEF eRisk 2018 challenge focuses on early detection of signs of depression or anorexia using posts or comments over social
media. The eRisk lab has organized two tasks this year and released two different corpora for the individual tasks. The corpora are developed using the posts and comments over Reddit, a popular social media.

Intially the corpara are realeased in XMl ile format. They are then processed into an csv file formate by extracting only Subject id and text if each individual users.The analysis and performance for early risk prediction of anorexia or depression involving  various classifiers with machine learning techniques and feature engineering schemes are well explained  in the paper( http://ceur-ws.org/Vol-2125/paper_182.pdf ). 

The DErisk.py  python file results the predicted output of the specifed classifer. 
The classifiers are ada boost, random forest, logistic regression and support vector machine are implemented with repective to  feature engineering schemes like bag of words(Count-Vectorizer, TF-IDF Vectorizer) , UMLS Metamap Features , and Embedding models  using Fasttext vector while implementing RNN's. This file only contains bag of word feature engennering thechnique. To evaluate results for UMLS metamap features change the read.csv path by importing Metamap resulted csv file.

UMLS Metamaps are the features extraced from the Metamap tool which extracts the medical concepts fromthe corpora which are related to depression. To follow more see the link https://metamap.nlm.nih.gov. To know more about the implementation of metamaps and other methodolgies you can refer to the above mentioned paper.The process_Metamap_output.py file shows how the extraction of featues are done.
This contains the process of extracting  required features from the corpora that helps in building a model that provied better predictions.


The implementation like Rnn and other futer works are explained in following paper (http://ceur-ws.org/Vol-2125/paper_182.pdf) 
Hence , For any quires can contact on jandhyyalasri@gmail.com.

Thanks & Regards,
J.Kalyani.

