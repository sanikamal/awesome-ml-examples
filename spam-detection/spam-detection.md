
# ML Application: Spam Detection

# What is Spam

SPAM can be defined as massive, undesired e-mail communications that are sent to large numbers of people
without their authorization. While the contents vary from one case to another, it has been
observed that the main topics of these mails are pharmacy products, gambling, weight loss,
and phishing attempts.

It is important to note that SPAM is not only annoying but also expensive. Today, many
people check their inboxes using a cell-phone data plan. Every e-mail requires an amount of
data transfer, which the client must pay for. Additionally, SPAM costs money for Internet
Service Providers (ISPs) as it is transmitted through their servers and other network
devices. Once we have considered this aspect of SPAM, we will want to avoid it to the
maximum extent possible.

# Import libraries


```python
# Import libraries
import csv
from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
```

Now load the dataset and print the total number of lines,each representing a record.
`rstrip()` Python method is used to strip whitespace characters from the end of each line.


```python
# Load the training dataset 'SMSSpamCollection' into variable 'messages'
messages = [line.rstrip() for line in open('sms-data/SMSSpamCollection')]
# Print number of messages
len(messages)
```




    5574



Inspect the messages by parsing the dataset file using pandas. The use of the `head()` method causes pandas to return only the first five rows


```python
# Read the dataset. Specify the field separator is a tab instead of a comma.
# Additionally, add column captions ('label' and 'message') for the two fields in the dataset.
# To preserve internal quotations in messages, use QUOTE_NONE.
messages = pd.read_csv('sms-data/SMSSpamCollection', sep='\t',quoting=csv.QUOTE_NONE,names=["class", "message"])
# Print first 5 records
messages.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



Note that the first column has been labeled class, whereas the second column is message. In class, we can see the
individual classification of each message as ham (good) or spam (bad):

Use the `groupby()` and `count()` methods to group the records by class and then count the number in each one.


```python
# Group by class and count
messages.groupby('class').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>message</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>4827</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>747</td>
    </tr>
  </tbody>
</table>
</div>



## The bag-of-words model
This is a common document classification technique where the occurrence and
the frequency of each word is used to train a classifier.


```python
#function to split each message into a series of words
def split_into_words(message):
    return TextBlob(message).words
```


```python
# This is what the first 5 records look when splitted into individual words
messages.message.head().apply(split_into_words)
```




    0    [Go, until, jurong, point, crazy, Available, o...
    1                       [Ok, lar, Joking, wif, u, oni]
    2    [Free, entry, in, 2, a, wkly, comp, to, win, F...
    3    [U, dun, say, so, early, hor, U, c, already, t...
    4    [Nah, I, do, n't, think, he, goes, to, usf, he...
    Name: message, dtype: object



Now Normalize the words into their base form and convert each message into a vector to train the model. 
In this step, words such as walking,walked, walks, and walk are reduced into their lemma–walk. Thus, the presence of
any of those words will actually count toward the number of occurrences of walk:


```python
# Convert each word into its base form
def words_into_base_form(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]
```


```python
# Convert each message into a vector
training_vector = CountVectorizer(analyzer=words_into_base_form).fit(messages['message'])
```


```python
# View occurrence of words in an arbitrary vector. Use 19 for vector #20.
message20 = training_vector.transform([messages['message'][19]])
message20
```




    <1x8859 sparse matrix of type '<class 'numpy.int64'>'
    	with 21 stored elements in Compressed Sparse Row format>




```python
# Print message #10 for comparison
print (messages['message'][9])
# Identify repeated words
print ('First word that appears twice:',training_vector.get_feature_names()[3437])
print ('Word that appears three times:',training_vector.get_feature_names()[5192])
```

    Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
    First word that appears twice: free2day
    Word that appears three times: model..sony
    

The term frequency (TF, the number of times a term occurs in a document) and inverse document frequency (IDF) of each word. The IDF diminishes the weight of a word that appears very frequently and increases the weight of words that do not occur often:


```python
# Bag-of-words for the entire training dataset
messagesBagOfWords = training_vector.transform(messages['message'])
# Weight of words in the entire training dataset -Term Frequency and Inverse Document Frequency
messagesTfidf = TfidfTransformer().fit(messagesBagOfWords).transform(messagesBagOfWords)
```

Based on these preceding statistical values, we will be able to train our modelusing the **Naive-Bayes algorithm**


```python
# Train the model
spamDetector = MultinomialNB().fit(messagesTfidf,messages['class'].values)
```

Congratulations! You have trained your model to perform SPAM detection. Now we'll test it against new data.


```python
# Test message
example = ["Thanks for your Ringtone Order, Reference T91. You will be charged GBP 4 per week. You can unsubscribe at anytime by calling customer services on 09057039994"]
# Result
checkResult = spamDetector.predict(training_vector.transform(example))[0]
print ('The message [',example[0],'] has been classified as', checkResult)
```

    The message [ Thanks for your Ringtone Order, Reference T91. You will be charged GBP 4 per week. You can unsubscribe at anytime by calling customer services on 09057039994 ] has been classified as spam
    


```python
# Test message
example = ["Can you say what happen"]
# Result
checkResult = spamDetector.predict(training_vector.transform(example))[0]
print ('The message [',example[0],'] has been classified as', checkResult)
```

    The message [ Can you say what happen ] has been classified as ham
    

As you can see, we have tested our model successfully with two test messages. Feel free to experiment with your own messages now.
