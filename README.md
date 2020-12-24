# Early Detection of Signs of Depression Over Social Media using Effective Machine Learning Frameworks :

The CLEF eRisk 2018 challenge focuses on early detection of signs of Depression using posts or comments over social
media. The eRisk lab has organized two tasks this year and released two different corpora for the individual tasks. The corpora are developed using the posts and comments over Reddit, a popular social media.Thisconsist of two classes postive and negitive whose data are released in sequential manner.

Initially the corpara are realeased in XMl file format. They are then processed into an csv file format by extracting only Subject_id , Classlabel and text of each individual users. Xml  ELement Tree library are used for pre-processing the text fro given xml files. The pre-processing method can be understood from pre-process.py file.

The classifiers like ada boost, random forest, logistic regression and support vector machine are implemented with repective to feature engineering schemes  bag of words(Count-Vectorizer, TF-IDF Vectorizer) , UMLS Metamap Features , and Embedding models  using Fasttext vector while implementing RNN's. The DErisk.py file only contains bag of word feature engennering thechnique.

UMLS Metamaps are the features extraced from the Metamap tool which extracts the medical concepts fromthe corpora which are related to depression. To follow more see the link https://metamap.nlm.nih.gov. To know more about the implementation of metamaps and other methodolgies you can refer to the above mentioned paper. The process_Metamap_output.py file shows how the extraction of featues are done.
This contains the process of extracting  required features from the corpora that helps in building a model that provied better predictions. The code in DErisk.py helps in training the model by simply chaningn the input of csv file.

The analysis and performance for prediction of signs depression over corpora are explained in the following paper with proper references.
( http://ceur-ws.org/Vol-2125/paper_182.pdf ) 

For any further query, comment or suggestion, you may contact Sree Kalyani at jandhyalasri@gmail.com


