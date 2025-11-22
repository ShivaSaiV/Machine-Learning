import re

def findUniqueWords(filename):
    f = open(filename, "r")
    s = f.read()
    count = 0
    s1 = re.sub(r'\W', " ", s)
    s1 = s1.lower()
    s1 = s1.split()
    unique_words = []
    for i in s1:
        if s1.count(i) == 1:
            unique_words.append(i)

    count = len(unique_words)

    unique_words = sorted(unique_words, reverse=True)
        

    return count, unique_words
    

# Example Test Cases:
# print(findUniqueWords("uniquewords.txt")) 


def getWords(sentence, letter):
    if len(sentence) == 0:
        return []
    
    if letter == "":
        return []
    
    words = []
    
    sentences = sentence.split()

    for s in sentences:
        if s[0].lower() == letter.lower() and s[len(s) - 1].lower() == letter.lower():
            if s.isalpha() == True:
                words.append(s)
    return words

# Example Test Cases:
s = "The TART program runs on Tuesday and Thursdays,but it does not start until next week."
print(getWords(s,"t"))   # -> ['TART']



def removeMultipleSpaces(string):
    str1 = re.sub(r"\s+", " ", string)
    return str1



print(removeMultipleSpaces("hello     world"))


def commaConverter(string):
    str1 = re.sub(r"\.", ",", string)
    str2 = list(str1)
    if str2[len(str2) - 1] == ",":
        str2[len(str2) - 1] = "!"
    str3 = "".join(str2)
    return str3

# Example Test Cases:
print(commaConverter(".hello.this.is.a.test.")) # -> ,hello,this,is,a,test!

def findSequence(string):
    string1 = re.sub(r"\W\s", " ", string)
    s1 = re.sub(r"\.", "", string1)
    s2 = s1.split()

    seq = []
    print(s2)
    for w in s2:
        if w[0].isupper() == True and w.isalnum() == True:
            seq.append(w)

    seq = sorted(seq)

    if len(seq) == 0:
        return None
    else:
        return seq

# Example Test Cases:
print(findSequence("We are studying the course Introduction to Data Science.")) # -> ['Data', 'Introduction', 'Science', 'We']
print(findSequence("today is our second lab assignment :P"))   # -> None
print(findSequence("B@gel Sandwich, Breakfast Pan!ni"))   # -> ['Breakfast', 'Sandwich']



def removeZeros(string):
    str1 = re.sub(r"\.0+(?=\d|\.|$)", ".", string)

    return str1

# Example Test Cases:
print(removeZeros("216.08.094.196")) # -> 216.8.94.196
# print(removeZeros("503.38.562.192")) # -> 503.38.562.192
# print(removeZeros("452.50.001.210")) # -> 452.50.1.210



def findStrings(string): 
    str1 = string.split()
    first_arr = []
    last_arr = []
    word_arr = []

    for i in str1:
        if i.endswith("ly") == True:
            first = string.find(i)
            add = len(i)
            last = first + add
            first_arr.append(first)
            last_arr.append(last)
            word_arr.append(i)

    ans = []

    for i in range(len(first_arr)):
        ans.append(f"{first_arr[i]}-{last_arr[i]}: {word_arr[i]}")

    str2 = ", ".join(ans)
    return str2

# Example Test Cases:    
print(findStrings("Clearly, he has no excuse for such behavior.")) #-> 0-7: Clearly
print(findStrings("The soldier fought bravely and strongly.")) #-> 19-26: bravely, 31-39: strongly
print(findStrings("The boy happily went to home and gladly completed his assignments.")) #-> 8-15: happily, 33-39: gladly