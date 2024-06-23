import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.pipeline import Pipeline

data = pd.read_csv('dataset.csv', encoding='latin-1')

label_encoder = LabelEncoder()
data['class_encoded'] = label_encoder.fit_transform(data['class'])

x = data['text']
y = data['class_encoded']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train = x_train.fillna('')
x_test = x_test.fillna('')

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('MCC:', mcc)

# new_text = """
# "Four-day work week, a remarkable idea to conserve energy and resources, some businesses have adopted the idea of a four-day work week, with each day consisting of ten hours. I think this idea is amazing, workers would have a day off and they won't be wasting energy and resources like in a regular day of work. Therefore i think my school should follow this model and extend the school day two hours in a four-day school week. Students would be happy of having a day off, and most of them would support the idea of having a four-day school week. By following this idea, my school would conserve energy and a lot of resources such as papers, school materials, food, etc.

# A four-day school week benefit students and teachers in many ways, students would have a day off and time to do extra curricular activities such as sports or work, teachers would be at home and have time to spend with their kids and be with the family, and all of us would be conserving energy and resources. One disadvantage of having a four-day school week is that some parents work 5 days a week in different companies and these companies might not have the four-day work week so they won't have time to be with their kids at home in the day off, and some of them would be forced to change their five-day work week to a four-day work week, so they can be with their kids at home.

# Personally, i would prefer the four-day school week, because,i have a part-time job after school , it is from 5 to 10pm, and having a day off would allow me to have time to do some extra work at home and to change my schedule for in the morning so i can have the rest of the day to work on my stuffs, have time to chill, or maybe to do my homework, and help my mom with whatever she needs.

# Having a four-day school week, would let me have time to do a lot of things during the day, for example, babysit my little brother if my mom goes out, go to the gym or maybe help my dad. I can get another job to work on the weekends, the thing is that it would allow me to be busy, and to generate money.

# In my family, my dad is the one who is always at work, he works from 8am to 6pm everyday, he doesn't have a day off, but it is okay for him. My mom, she doesn't work, she is always at home taking care of me and my little brother, for my parents, it wouldn't be a problem for me to have four days of school a week, because my mom would take care of me during that day off.

# In conclusion, i think having a four-day school week, would benefit a lot of people, because just like me, there are plenty of students that would love to work more and generate money, and for the teachers, well they would enjoy have a day off and spend time with their kids at home."
# """

# new_text_transformed = pipeline.named_steps['vectorizer'].transform([new_text])

# predicted_class = pipeline.named_steps['classifier'].predict(new_text_transformed)

# predicted_class_name = label_encoder.inverse_transform(predicted_class)

# print('Predicted class:', predicted_class_name[0])
