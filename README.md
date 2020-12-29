# Early Detection of Anorexia and Depression Over Social Media using NLP and Machine Learning

The [CLEF eRisk 2018](https://early.irlab.org/2018/index.html) NLP shared task had focused on early detection of signs of anorexia or depression over posts or comments on Reddit. The eRisk lab had organized two different corpora for the individual tasks. The corpora were developed using the posts and comments over Reddit, a popular social media. The data consist of two classes, namely, postive and negitive, which were released in sequential manner over a period of time.

The data were realeased in XMl format, which were processed into a csv file by extracting the given subject id, class label and text of individual users. [XML ELement Tree](https://docs.python.org/3/library/xml.etree.elementtree.html) library in python was used to get free text data from the given XML files. The preprocessing of the given depression data was performed by `preprocess_depression.py` and the preprocessing of anorexia data was implemented within `anorexia.py`. 

The classifiers viz., `ada boost`, `random forest`, `logistic regression` and `support vector machine` were implemented in terms of bag of words ([TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)) and [UMLS](https://metamap.nlm.nih.gov/) (a biomedical ontology) features. Moreover, recurrent neural network in terms of pretrained Fasttext and Glove word-embeddings was also implemented. The `anorexia.py` and `depression.py` files perform data classification using bag of words model respectively for anorexia and depression. The performance of data classification in terms of UMLS features were poor as there were very few biomedical terms in the data. Hence we did not upload these codes here. 

The analysis and performance of the frameowrks for the individual tasks are presented in this paper: http://ceur-ws.org/Vol-2125/paper_182.pdf  

For any further query, comment or suggestion, you may contact Sayanta Paul at sayanta95@gmail.com or Sree Kalyani at jandhyalasri@gmail.com or Tanmay Basu at welcometanmay@gmail.com


