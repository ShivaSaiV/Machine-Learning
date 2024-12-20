# Write a function to sum all the values in a dictionary.

# sumValues(dictionary) takes as input a dictionary (with integer values), and returns the sum of the  
# values for all keys in the dictionary!
#
# Example: 
#   print( sumValues({"a" : 7, "b" : 10}) ) # outputs 17

def sumValues(dictionary):
    sum = 0
    for i in dictionary:
        sum += dictionary[i]
        
    return sum



# Write a function to find the key corresponding to the maximum value in a dictionary. 

# maxValue(dictionary) takes as input a dictionary (with integer values), and returns the key 
# associated with the maximum value in the dictionary. 
#
# Example: 
#   print( maxValue({"a" : 7, "b" : 10}) ) # outputs b

def maxValue(dictionary):
    max_val = -99999999
    max_element = ""

    for i in dictionary:
        if dictionary[i] > max_val:
            max_val = dictionary[i]
            max_element = i



    return max_element

print(maxValue({"a" : 7, "b" : 10}))


def numberOfUniqueElements(listOfElements):
    count = 0
    arr1 = []
    for i in listOfElements:
        if i not in arr1:
            arr1.append(i)

    count = len(arr1)

    return count

print( numberOfUniqueElements([1,2,2,'c']) )



def splitStringCount(string):
    count = 0

    str1 = string.split()

    count = len(str1)

    return count



def isPalindrome(string):
    if string == "":
        return False
    
    str2 = string[::-1]

    return string.lower() == str2.lower()

print( isPalindrome("banana") )



def uniquePalindromes(string):
    
    str1 = string.lower()

    str2 = str1.split()

    arr1 = []
    
    for i in str2:
        if isPalindrome(i) == True:
            arr1.append(i)
            
    ans1 = []
    for i in arr1:
        if i not in ans1:
            ans1.append(i)

    arr2 = sorted(ans1, reverse=True)
    
    if len(arr2) == 0:
        return []
    else:
        return arr2

print(uniquePalindromes("sedf is like"))
print( uniquePalindromes('Madam asked Anna to go out but Anna refused') )


def numWords(filename):
    f = open(filename, "r")

    arr = f.readlines()
    num_words = []

    for i in arr:
        arr1 = i.split()
        count = len(arr1)
        num_words.append(count)

    return num_words



def averageWords(filename):
    f = open(filename, "r")

    sum = 0

    arr = f.readlines()
    leng = len(arr)
    
    for i in arr:
        arr1 = i.split()
        sum += len(arr1)
    
    avg = sum / leng

    return avg


def countLines(filename):
    sum = 0

    f = open(filename, "r")

    arr = f.readlines()

    for i in arr:
        if "ing" in i:
            sum += 1

    return sum




# Write a function to read text from an input file, find all unique palindromes and return them in 
# sorted order. 

# findPalindromes(filename) takes as input the file to read the text from, and returns a list of the 
# unique palindromes sorted in decreasing lexicographic order (reverse sorted order), in lower case. 
# You can use your code above for checking if a string is a palindrome.

# Note: In Q10, do not strip characters. Split only on whitespace, some palindromes may be outside the alphanumeric set.

# For example,
# filename: palindrome_test.txt
# output: ['tattarrattat', 'redivider', 'detartrated', 'aibohphobia', 'a']


def findPalindromes(filename):
    f = open(filename, "r")

    str1 = f.read()

    str1 = str1.lower()

    str2 = str1.split()


    arr1 = []
    for i in str2:
        i = i.replace(".", "")
        print(i)
        if (i == i[::-1]):
            arr1.append(i)
    
    arr2 = []
    for i in arr1:
        if i not in arr2:
            arr2.append(i)

    arr2 = sorted(arr2, reverse=True)

    return arr2


print(findPalindromes("/Users/shivasaivummaji/Desktop/CS:DS/Semesters/Fall 2023/CS 242/palindrome.txt"))
