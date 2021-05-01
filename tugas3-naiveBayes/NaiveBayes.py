# In[]
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# In[]
d = pd.read_csv('Mall_Customers.csv', sep =',')

# In[]
d['Gender'] = d.apply(lambda row: 1 if (row['Gender']) == 'Female' else 0, axis=1)
d = d.drop(['CustomerID'], axis = 1)

# In[]
d_attribute = d.iloc[:, :3]
d_label = d.iloc[:, 3:]


# In[]
clf = GaussianNB()

# In[]
fitting = clf.fit(d_attribute, d_label.values.ravel())
prediksi = clf.predict(d_attribute)

# In[]

# Cek akurasi
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(d_label, prediksi))

# In[]

# Penulisan input [Jenis Kelamin, Umur, Pendapatan ($)]
# Jenis Kelamin 0 = Female, 1 = Male

prediction_input = clf.predict([[1, 21, 113]])

print (prediction_input)
