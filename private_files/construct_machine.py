import pandas

from sklearn.ensemble import RandomForestClassifier

import pickle

dataset = pandas.read_csv("private_dataset.csv")

## use iloc to extract column
## specify all rows and the column number (this is column 31 since index starts at 0)
target = dataset.iloc[:,30].values
## use iloc to extract multiple cols, 0 is inclusive, 30 is not inclusive
data = dataset.iloc[:,0:30].values


machine = RandomForestClassifier(criterion='gini', max_depth=10, n_estimators=11)
machine.fit(data,target)

with open("machine.pickle","wb") as file: ## or .pkl
	pickle.dump(machine,file)



