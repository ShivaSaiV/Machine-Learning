import pandas as pd
import numpy as np
import math

def medianMovieRating(filename):
    df = pd.read_csv(filename)
    rotten = df["Rotten Tomatoes Review"]
    imdb = df["IMDb Review"]
    meta = df["Metacritic Review"]
    google = df["Google Review"]

    rotten_med = rotten.median()
    imdb_med = imdb.median()
    meta_med = meta.median()
    google_med = google.median()

    Dict = {"Rotten Tomatoes Review": rotten_med, "IMDb Review": imdb_med, "Metacritic Review": meta_med, "Google Review": google_med}


    return Dict

print(medianMovieRating("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/movie_reviews.csv"))  
# -> {'Rotten Tomatoes Review': 0.93, 'IMDb Review': 0.85, 'Metacritic Review': 0.87, 'Google Review': 0.93}


def meanRating(filename):
    df = pd.read_csv(filename)

    
    df["Mean Rating"] = (df.iloc[:, 2:].mean(axis=1))

    return df

# Example Output
print(meanRating("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/movie_reviews.csv"))


def sortMean(filename):

    df = pd.read_csv(filename)
    df = meanRating(filename)
    df = df.sort_values(by=["Mean Rating"], ascending=False)
    names = list(df["Movie"])
    

    return names

    
# Example Output
print(sortMean("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/movie_reviews.csv"))



def rubric(value):
    grade = ""
    if (value >= 0.9):
        grade = "A"
    elif value >= 0.8:
        grade = "B"
    elif value >= 0.7:
        grade = "C"
    else:
        grade = "D"
    
    
    return grade
    

# Examples
print(rubric(.95))  # -> 'A'
print(rubric(.80)) # -> 'B'


def assignGrade(filename):
    df = pd.read_csv(filename)
    
    l1 = list(df["IMDb Review"].apply(rubric))

    return l1
    
# Example
print(assignGrade("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/movie_reviews.csv"))
# -> ['C', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'C', 'B']



def findAGrades(filename):

    df = pd.read_csv(filename)
    
    l1 = list(df["Rotten Tomatoes Review"].apply(rubric))
    l2 = list(df["IMDb Review"].apply(rubric))
    l3 = list(df["Metacritic Review"].apply(rubric))
    l4 = list(df["Google Review"].apply(rubric))

    n1 = l1.count("A")
    n2 = l2.count("A")
    n3 = l3.count("A")
    n4 = l4.count("A")

    nums = [n1, n2, n3, n4]

    return nums

# Example
print(findAGrades("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/movie_reviews.csv"))
# -> [10, 2, 5, 12]