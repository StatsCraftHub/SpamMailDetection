```python
#This program detects if an email is spam or not
```


```python
#import libraries

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
```


```python
import pandas as pd

# Try different encodings until you find the correct one
encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'utf-16']

file_path = 'D:\\Anju\\Projects\\Spam mail detection\\spam.csv'

for encoding in encodings_to_try:
    try:
        df = pd.read_csv(file_path, encoding=encoding, delimiter=',')
        break  # If successful, exit the loop
    except UnicodeDecodeError:
        continue  # If decoding fails, try the next encoding

# Display the contents of the DataFrame
df
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
      <th>Spa</th>
      <th>text</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 5 columns</p>
</div>




```python
# drop the variables
df = df.iloc[:, :2]
df
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
      <th>Spa</th>
      <th>text</th>
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
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 2 columns</p>
</div>




```python
# Assuming df is your DataFrame with a "Spa" column
df = df.copy()  # Create a copy of the DataFrame
df['Spamm'] = df['Spa'].apply(lambda x: 0 if x == 'ham' else 1)
df
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
      <th>Spa</th>
      <th>text</th>
      <th>Spamm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>spam</td>
      <td>This is the 2nd time we have tried 2 contact u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>ham</td>
      <td>Will Ì_ b going to esplanade fr home?</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>ham</td>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>ham</td>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>ham</td>
      <td>Rofl. Its true to its name</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 3 columns</p>
</div>




```python
#Get the shape (Get the number of raws and columns)

df.shape

```




    (5572, 3)




```python
# drop the first column (index 0)
df = df.drop(columns=df.columns[:1])
df
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
      <th>text</th>
      <th>Spamm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ok lar... Joking wif u oni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>U dun say so early hor... U c already then say...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5567</th>
      <td>This is the 2nd time we have tried 2 contact u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>Will Ì_ b going to esplanade fr home?</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5569</th>
      <td>Pity, * was in mood for that. So...any other s...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5570</th>
      <td>The guy did some bitching but I acted like i'd...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5571</th>
      <td>Rofl. Its true to its name</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5572 rows × 2 columns</p>
</div>




```python
#Get the column names

df.columns
```




    Index(['text', 'Spamm'], dtype='object')




```python
#Check for duplicates and remove them
df.drop_duplicates(inplace = True)
```


```python
#Get the new shape of the dataset
df.shape
```




    (5169, 2)




```python
#show the number of missing data(NAN, NaN, na) for each column
df.isnull().sum()
```




    text     0
    Spamm    0
    dtype: int64




```python
#Download stopwords package

nltk.download('stopwords')

```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Jibin\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\stopwords.zip.
    




    True




```python
import string
from nltk.corpus import stopwords

# Create a function to process the text
def process_text(text): 
    # Remove punctuation from the text
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    # Remove stopwords
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    # Return a list of clean text words
    return clean_words

# Apply the process_text function to the 'text' column
df['text'].head().apply(process_text)

```




    0    [Go, jurong, point, crazy, Available, bugis, n...
    1                       [Ok, lar, Joking, wif, u, oni]
    2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...
    3        [U, dun, say, early, hor, U, c, already, say]
    4    [Nah, dont, think, goes, usf, lives, around, t...
    Name: text, dtype: object




```python
#convert a collection of text to a matriz of tokens

from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])
```


```python
#Split the data into 80% and 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(messages_bow, df['Spamm'], test_size=0.20, random_state=0)

```


```python
#Get the shape of messages_bow

messages_bow.shape
```




    (5169, 11304)




```python
#Create and train the Naive Bayes Classifier(multyinomial naive calssifier which is suitable for clasification with descrete features)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train,Y_train)
```


```python
#print the predictions
print(classifier.predict(X_train))
```

    [0 0 0 ... 0 0 0]
    


```python
#print the actual values

print(Y_train.values)
```

    [0 0 0 ... 0 0 0]
    


```python
#Evaluate the model on training dataset

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(X_train)
print(classification_report(Y_train, pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00      3631
               1       0.98      0.98      0.98       504
    
        accuracy                           1.00      4135
       macro avg       0.99      0.99      0.99      4135
    weighted avg       1.00      1.00      1.00      4135
    
    


```python

print('confusion_matrix: \n', confusion_matrix(Y_train, pred))
```

    confusion_matrix: 
     [[3623    8]
     [  11  493]]
    


```python
print('Accuracy: ', accuracy_score(Y_train, pred))
```

    Accuracy:  0.9954050785973397
    


```python
#print the predictions
print(classifier.predict(X_test))

#print the actual values
print(Y_test.values)
```

    [0 0 0 ... 0 0 0]
    [0 0 0 ... 0 0 0]
    


```python
#Evaluate the model on test dataset

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
pred = classifier.predict(X_test)
print(classification_report(Y_test, pred))

print('confusion_matrix: \n', confusion_matrix(Y_test, pred))

print('Accuracy: ', accuracy_score(Y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.96      0.97       885
               1       0.80      0.93      0.86       149
    
        accuracy                           0.96      1034
       macro avg       0.89      0.94      0.92      1034
    weighted avg       0.96      0.96      0.96      1034
    
    confusion_matrix: 
     [[850  35]
     [ 11 138]]
    Accuracy:  0.9555125725338491
    


```python

```
