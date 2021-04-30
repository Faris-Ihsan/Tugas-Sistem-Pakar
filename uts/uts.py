# In[]
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# In[]
d = pd.read_csv('dataset_uts.csv', sep =';')

d_attribute = d.iloc[:, :3]
d_label = d.iloc[:, 3:]


# In[]
clf = GaussianNB()

# In[]
fitting = clf.fit(d_attribute, d_label.values.ravel())

prediksi = clf.predict(d_attribute)

# In[]
prediction_input = clf.predict([[4, 40, 100]])

print (prediction_input)


# In[]
cm = confusion_matrix(d_label, prediksi)

cm
