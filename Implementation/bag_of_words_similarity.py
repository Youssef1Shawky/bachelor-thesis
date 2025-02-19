import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import tkinter as tk


# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in words]
    preprocessed_text = ' '.join(lemmas)
    return preprocessed_text

def calculate_similarity(text1, text2):
    preprocessed_text1 = preprocess_text(text1)
    preprocessed_text2 = preprocess_text(text2)
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])
    similarity = cosine_similarity(bow_matrix[0], bow_matrix[1])[0][0]
    return similarity


def calculate_average_similarity(text1,col):
    df = pd.read_csv('daigt_external_dataset.csv')
    similarity_scores = []
    for text2 in df[col]:
        similarity_score = calculate_similarity(text1, text2)
        similarity_scores.append(similarity_score)
        if len(similarity_scores) % 100 == 0:
            print(f"Processed {len(similarity_scores)*10} texts")
        if len(similarity_scores) == 200:
            break
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    return average_similarity




# df = pd.read_csv('AI_Text.csv', encoding='latin-1')
# cnt = 0
# TP = 0
# FN = 0
# for text in df['text']:
#     average_similarity1 = calculate_average_similarity(text, 'text')
#     average_similarity2 = calculate_average_similarity(text, 'source_text')
#     cnt += 1
#     print(f"done with {cnt} texts")
#     if average_similarity1 > average_similarity2:
#         FN += 1
#     else:
#         TP += 1
#     print(f"TP: {TP}, FN: {FN}")


# df = pd.read_csv('Human_text.csv', encoding='latin-1')
# cnt = 0
# TN = 0
# FP = 0
# for text in df['text']:
#     average_similarity1 = calculate_average_similarity(text, 'text')
#     average_similarity2 = calculate_average_similarity(text, 'source_text')
#     cnt += 1
#     print(f"done with {cnt} texts")
#     # print(f"average_similarity1: {average_similarity1}, average_similarity2: {average_similarity2}")
#     if average_similarity1 > average_similarity2:
#         TN += 1
#     else:
#         FP += 1
#     print(f"TN: {TN}, FP: {FP}")


TP = 63
FN = 37
TN = 100;
FP = 0

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * precision * recall / (precision + recall)
MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")
print(f"MCC: {MCC}")























# def calculate_button_clicked():
#     text1 = text_input.get("1.0", tk.END).strip()
#     if text1:
#         average_similarity1 = calculate_average_similarity(text1, 'text')
#         average_similarity2 = calculate_average_similarity(text1, 'source_text')
#         if average_similarity1 > average_similarity2:
#             result_label.config(text=f"The text is more likely to be written by a human" , fg="blue")
#         else:
#             result_label.config(text=f"The text is more likely to be generated by an AI model", fg="red")

# window = tk.Tk()
# window.title("AI Detector")
# window.geometry("600x400")
# frame = tk.Frame(window, pady=10) # Decreased the pady value to add less vertical space
# frame.pack()
# text_label = tk.Label(frame, text="Enter your text:", font=("Arial", 14))
# text_label.pack()
# text_input = tk.Text(frame, height=10, width=50, font=("Arial", 12), bd=2, relief=tk.SOLID, bg="white", fg="black")
# text_input.pack()
# button_frame = tk.Frame(window)
# button_frame.pack()
# calculate_button = tk.Button(button_frame, text="Detect Text", command=calculate_button_clicked, font=("Arial", 12),
#                               bg="light sky blue", activebackground="blue", padx=10, pady=5)
# calculate_button.pack()
# result_frame = tk.Frame(window, pady=20)
# result_frame.pack()
# result_label = tk.Label(result_frame, text="", font=("Arial", 16))
# result_label.pack()
# window.mainloop()