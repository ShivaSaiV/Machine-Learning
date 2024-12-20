import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def tokenization(book):
    f = open(book, "r")
    text = f.read()
    text = text.lower()
    text = re.sub(r'[^\w\s]|', "", text)
    text = re.sub(r'[_=\.\-\d+]', "", text)
    text = re.sub(r'\d+', "", text)
    text = re.sub(r'\b(?:i|the|we|our|you|him|me|she|her|x|y|it|them|their|this|that|these|those|he|an|a|us|no|in|on|but|to|of|be|is)\b', "", text)

    list_words = text.split()

    return list_words

tokenized_biology = tokenization("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/Project 1/biology physiology.txt")
tokenized_ibm = tokenization("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/Project 1/ibm programming systems.txt")
tokenized_livingmachine = tokenization("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/Project 1/the story of living machine.txt")
tokenized_hacking = tokenization("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/Project 1/underground hacking.txt")

book_names = ["The Story of the Living Machine", "IBM 1401 Programming Systems", "The Biology, Physiology, and Sociology of Reproduction", 
                  "Underground: Hacking, madness and obsession on electronic frontier"]

list_of_all_words = list(set(tokenized_biology + tokenized_ibm + tokenized_livingmachine + tokenized_hacking))

def word_document_table():
    df = pd.DataFrame(columns=book_names)
    
    for w in list_of_all_words:
        bio = tokenized_biology.count(w)
        ibm = tokenized_ibm.count(w)
        living_machine = tokenized_livingmachine.count(w)
        hack = tokenized_hacking.count(w)

        df.loc[w] = [bio, ibm, living_machine, hack]
    
    df = df.transpose()
    
    return df

df = word_document_table()
print(df)
# df.to_csv("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/word_document_table.csv", index=False)


list_books = [tokenized_biology, tokenized_ibm, tokenized_livingmachine, tokenized_hacking]

dict_bio = {}
for w in tokenized_biology:
    tf = tokenized_biology.count(w) / len(tokenized_biology)
    dict_bio[w] = (w, tf)

dict_ibm = {}
for w in tokenized_ibm:
    tf = tokenized_ibm.count(w) / len(tokenized_ibm)
    dict_ibm[w] = (w, tf)

dict_livingmachine = {}
for w in tokenized_livingmachine:
    tf = tokenized_livingmachine.count(w) / len(tokenized_livingmachine)
    dict_livingmachine[w] = (w, tf)

dict_hacking = {}
for w in tokenized_hacking:
    tf = tokenized_hacking.count(w) / len(tokenized_hacking)
    dict_hacking[w] = (w, tf)

dict_bio = dict(sorted(dict_bio.items(), key=lambda item: item[1]))
dict_ibm = dict(sorted(dict_ibm.items(), key=lambda item: item[1]))
dict_livingmachine = dict(sorted(dict_livingmachine.items(), key=lambda item: item[1]))
dict_hacking = dict(sorted(dict_hacking.items(), key=lambda item: item[1]))

dict_idf_bio = {}
dict_idf_ibm = {}
dict_idf_livingmachine = {}
dict_idf_hacking = {}

for w in list_of_all_words:
    counter = 0
    if w in tokenized_biology:
        counter += 1
    
    if w in tokenized_ibm:
        counter += 1
    
    if w in tokenized_livingmachine:
        counter += 1
        
    if w in tokenized_hacking:
        counter += 1

    if counter == 0:
        counter = 1
    else:
        dict_idf_bio[w] = np.log(4 / (counter))
        dict_idf_ibm[w] = np.log(4 / (counter))
        dict_idf_livingmachine[w] = np.log(4 / (counter))
        dict_idf_hacking[w] = np.log(4 / (counter))

dict_idf_bio = dict(sorted(dict_idf_bio.items(), key=lambda item: item[1]))
dict_idf_ibm = dict(sorted(dict_idf_ibm.items(), key=lambda item: item[1]))
dict_idf_livingmachine = dict(sorted(dict_idf_livingmachine.items(), key=lambda item: item[1]))
dict_idf_hacking = dict(sorted(dict_idf_hacking.items(), key=lambda item: item[1]))

def get_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = {}
    for word in tf_dict:
        tf_val = tf_dict[word][1] 
        idf_val = idf_dict[word]
        tf_idf = tf_val * idf_val
        tf_idf_dict[word] = tf_idf
    
    return tf_idf_dict

dict_tf_idf_bio = get_tf_idf(dict_bio, dict_idf_bio)
dict_tf_idf_ibm = get_tf_idf(dict_ibm, dict_idf_ibm)
dict_tf_idf_livingmachine = get_tf_idf(dict_livingmachine, dict_idf_livingmachine)
dict_tf_idf_hacking = get_tf_idf(dict_hacking, dict_idf_hacking)


dict_tf_idf_bio = dict(sorted(dict_tf_idf_bio.items(), key=lambda item: item[1], reverse=True))
dict_tf_idf_ibm = dict(sorted(dict_tf_idf_ibm.items(), key=lambda item: item[1], reverse=True))
dict_tf_idf_livingmachine = dict(sorted(dict_tf_idf_livingmachine.items(), key=lambda item: item[1], reverse=True))
dict_tf_idf_hacking = dict(sorted(dict_tf_idf_hacking.items(), key=lambda item: item[1], reverse=True))


def get_top_3(tf_idf_dict):
    first_3 = {}
    tf_idf_dict = list(tf_idf_dict.items())
    for i in range(3):
        first_3[tf_idf_dict[i][0]] = tf_idf_dict[i][1]
    return first_3

top_3_bio = get_top_3(dict_tf_idf_bio)
top_3_ibm = get_top_3(dict_tf_idf_ibm)
top_3_livingmachine = get_top_3(dict_tf_idf_livingmachine)
top_3_hacking = get_top_3(dict_tf_idf_hacking)

print("Top 3 for Bio Book" + str(top_3_bio))
print("Top 3 for IBM Book" + str(top_3_ibm))
print("Top 3 for Living Machine Book" + str(top_3_livingmachine))
print("Top 3 for Hacking Book" + str(top_3_hacking))


def plot_top3_tfidf(top3_tfidf_dict, name):
    words = list(top3_tfidf_dict.keys())
    tfidf_values = list(top3_tfidf_dict.values())

    plt.figure(figsize=(10, 6))
    plt.barh(words, tfidf_values, color='blue')
    plt.xlabel('TF-IDF Value')
    plt.ylabel('Words')
    plt.title(f'Top TF-IDF Words for {name}')
    plt.show()

plot_top3_tfidf(top_3_bio, "Biology Book")
plot_top3_tfidf(top_3_ibm, "IBM Book")
plot_top3_tfidf(top_3_livingmachine, "Living Machine Book")
plot_top3_tfidf(top_3_hacking, "Hacking Book")



