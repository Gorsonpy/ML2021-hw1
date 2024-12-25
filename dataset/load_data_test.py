import csv

train_path = 'data/covid.train.csv'
test_path = 'data/covid.test.csv'

with open(train_path, 'r') as f:
    data = list(csv.reader(f))
    print(data[:5])