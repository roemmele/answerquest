import json
import pandas as pd
import cPickle as pickle
import os

_dir_name = 'maluuba/newsqa'
combined_dataset = json.load(open('combined-newsqa-data-v1.json'))['data']

train_story_ids = set(pd.read_csv(os.path.join(_dir_name, 'train_story_ids.csv'))['story_id'].values)
dev_story_ids = set(pd.read_csv(os.path.join(_dir_name, 'dev_story_ids.csv'))['story_id'].values)
test_story_ids = set(pd.read_csv(os.path.join(_dir_name, 'test_story_ids.csv'))['story_id'].values)

train_data_jsons = []
dev_data_jsons = []
test_data_jsons = []

for d in combined_dataset:
    id = d['storyId']
    if id in train_story_ids:
        train_data_jsons.append(d)
    elif id in dev_story_ids:
        dev_data_jsons.append(d)
    elif id in test_story_ids:
        test_data_jsons.append(d)

print ("Training set: ", len(train_data_jsons))
print ("Dev set: ", len(dev_data_jsons))
print ("Test set: ", len(test_data_jsons))

pickle.dump(train_data_jsons, open('train_data.pkl', 'wb'))
pickle.dump(dev_data_jsons, open('dev_data.pkl', 'wb'))
pickle.dump(test_data_jsons, open('test_data.pkl', 'wb'))
