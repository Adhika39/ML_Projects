import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import LSTM,Dense,Bidirectional,Dropout,Conv1D,MaxPooling1D,Embedding,Flatten
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt

def visualize_model(model):
    plt.figure(figsize=(12, 6))
    layer_names = [layer.name for layer in model.layers]
    for i, layer in enumerate(layer_names):
        plt.text(0, -i, layer, ha='center', fontsize=12)
    plt.xlim(-1, 1)
    plt.ylim(-len(layer_names), 1)
    plt.axis('off')
    plt.title('Model Architecture')
    plt.show()




df = pd.read_csv('FinancialPhraseBank.csv', encoding='latin1')
print("df.head() ", df.head())
print("df.tail() ", df.tail())
print("df.info() ", df.info())
print("df.nunique()", df.nunique())
status = df.iloc[:,0]
text = df.iloc[:,1]
print("status.unique()",status.unique())
print("status.value_Counts()",status.value_counts())
# print("df.status.unique() ", df.status.unique())
# print("df.status.value_counts() ", df.status.value_counts())

# print("df.isnull().sum() ", df.isnull().sum())
#
# df.text = df.text.fillna(df.text.mode()[0])
# print("df.isnull().sum() ", df.isnull().sum())
#
# print("df.columns ", df.columns)
# df.drop('Unnamed: 0' , axis = 1,inplace = True)
# print("df.columns ", df.columns)


stopwordss = stopwords.words('english')
print("stopwordss ", stopwordss)
df['status'].isnull().sum()

stopwordss = stopwords.words('english')
lem = WordNetLemmatizer()

def clean(line):
    line = line.lower()
    line = re.sub(r'\d+', '', line)
    line = re.sub(r'[^a-zA-Z0-9\s]', '', line)
    translator = str.maketrans('', '', string.punctuation)
    line = line.translate(translator)
    words = [word for word in line.split() if word not in stopwordss]
    #     words = [lem.lemmatize(word) for word in words]

    return ' '.join(words)

text = text.apply(clean)
print("df.tail() ", df.tail())

for g in text:
    maxx = g.split()
    m = max([len(maxx)])

print("m ", m)

le = LabelEncoder()
status=le.fit_transform(status)
print(list(le.classes_))
print(le.transform(['neutral', 'positive', 'negative']))

# Model
x =text
y = status

tokenizer = Tokenizer(oov_token='<unk>',num_words=2500)
tokenizer.fit_on_texts(x.values)
data_x = tokenizer.texts_to_sequences(x.values)

vocab = tokenizer.word_index
l_voc = len(vocab)
print(l_voc)

em_sz = 70
pad_sz = 42
latent_sz = 300
data_x = pad_sequences(data_x,maxlen=pad_sz,padding = 'post',truncating = 'post')

kernel_size = 5
filters = 128
pool_size = 4

optimizer = Adam(learning_rate = 0.0005)

model = Sequential()
model.add(Embedding(input_dim=l_voc, output_dim=em_sz))
model.add(Dropout(0.5))

model.add(Conv1D(filters,kernel_size,padding = 'valid',activation = 'relu',strides = 1))
# model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size = pool_size))
model.add(LSTM(latent_sz))
model.add(Flatten())
model.add(Dense(3,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.build((None , pad_sz))
model.summary()

plot_model(model, show_shapes=True,show_layer_names=True,dpi = 90, to_file='model_plot.png')

y= pd.get_dummies(y).values
x_train,x_test,y_train,y_test = train_test_split(data_x,y,random_state=0,shuffle = True,stratify=y , test_size = .2)
history = model.fit(x_train,y_train,batch_size=50, epochs=10,validation_data=(x_test,y_test))

score, acc = model.evaluate(x_test, y_test)
print(score)
print(acc)

p = model.predict(x_test)
print("p = ", p)

pred = np.argmax(model.predict(x_test[7:8]))
print("pred ", pred)
print("y_test[7:8] ", y_test[7:8])

prediction = le.inverse_transform(pred.reshape(1))[0]
print("prediction ", prediction)

history_dict = history.history
history_dict.keys()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()

plt.show()
