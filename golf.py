from sklearn import tree
import pandas as pd

#create empty data frame
golf_df = pd.DataFrame()

#add outlook
golf_df['Outlook'] = ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy',
                      'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast',
                      'overcast', 'rainy']

#add temperature
golf_df['Temperature'] = [92, 86, 83, 70, 60, 53, 62,
                          75, 57, 72, 78, 69, 81, 71]

#add humidity
golf_df['Humidity'] = [40.3, 63.1, 35.3, 86.3, 72.4, 22.7, 27.0,
                       80.7, 15.4, 25.1, 18.6, 58.3, 27.2, 39.3]

#add windy
golf_df['Windy'] = ['false', 'true', 'false', 'false', 'false', 'true', 'true',
                    'false', 'false', 'false', 'true', 'true', 'false', 'true']

#finally add play
golf_df['Play'] = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes',
                   'yes', 'yes', 'no']

golf_df['weekday'] = ['yes', 'yes', 'no', 'yes', 'yes', 'no',
                      'yes', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes']

golf_df.head(10)

print(golf_df.head(3))
print(golf_df)

one_hot_data = pd.get_dummies(
    golf_df[['Outlook', 'Windy', 'weekday']])  # get_dummies
one_hot_data

# join the continuous data with one hot data using pandas .join()
golf_one_hot = golf_df[['Temperature', 'Humidity']].join(one_hot_data)
golf_one_hot

clf = tree.DecisionTreeClassifier()

clf_train = clf.fit(golf_one_hot, golf_df['Play'])

clf_train

print(tree.export_graphviz(clf_train, None))

prediction = clf_train.predict([[62, 15.3, 0, 0, 1, 0, 1, 1, 0]])

prediction

print(prediction)
