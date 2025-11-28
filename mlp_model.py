import glob
import os
import librosa
import time
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

tess_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


def extract_feature(file_name):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    result = np.array([])

    stft = np.abs(librosa.stft(X))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chromas))

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    result = np.hstack((result, mfccs))

    mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
    result = np.hstack((result, mels))

    return result


data = pd.read_csv("./TESS_FEATURES.csv")


print(data.head())

data.shape

#printing all columns
data.columns

#dropping the column Unnamed: 0 to removed shuffled index
data = data.drop('Unnamed: 0',axis=1)
data.columns

#separating features and target outputs
X = data.drop('emotion', axis = 1).values
y = data['emotion'].values
print(y)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X.shape, y.shape

np.unique(y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)
print("mlp   *************************************************************************")
from sklearn.neural_network import MLPClassifier

steps3 = [('scaler', StandardScaler()),
          ('MLP', MLPClassifier())]

pipeline_mlp = Pipeline(steps3)

mlp = pipeline_mlp.fit(X_train, y_train)

print('Accuracy with Scaling: {}'.format(mlp.score(X_test, y_test)))

mlp_train_acc = float(mlp.score(X_train, y_train)*100)
print("----train accuracy score %s ----" % mlp_train_acc)

mlp_test_acc = float(mlp.score(X_test, y_test)*100)
print("----test accuracy score %s ----" % mlp_train_acc)

mlp_res = cross_val_score(mlp, X, y, cv=cv, n_jobs=-1)
print(mlp_res)
print("Average:", np.average(mlp_res))

mlp_pred = mlp.predict(X_test)
print(mlp_pred)

print(classification_report(y_test,mlp_pred))

acc_mlp = float(accuracy_score(y_test,mlp_pred))*100
print("----accuracy score %s ----" % acc_mlp)

cm_mlp = confusion_matrix(y_test,mlp_pred)

ax= plt.subplot()
sns.heatmap(cm_mlp, annot=True, fmt='g', ax=ax);

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix (Multi Layer Perceptron)');
ax.xaxis.set_ticklabels(tess_emotions);
ax.yaxis.set_ticklabels(tess_emotions);
plt.show()

import pickle

# Save MLP model in a pickle file
with open('mlp.pkl', 'wb') as f:
    pickle.dump(mlp, f)

import librosa
import numpy as np
import pandas as pd
import joblib

import pickle

# Open the PKL file in binary mode
with open('mlp.pkl', 'rb') as f:
    # Load the object from the file
    model = pickle.load(f)
