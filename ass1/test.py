import pandas as pd
import numpy as np
print("start to read")
ratings = pd.read_csv('./dataSet/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
ratings.columns = ['userId', 'ISBN', 'bookRating']

print(ratings.head(5))

users = pd.read_csv('./dataSet/BX-Users.csv', delimiter=";", encoding="latin1")
users.columns = ['userId', 'location', 'age']
print(users.head())


books = pd.read_csv('./dataSet/BX-Books.csv', delimiter=";", encoding="latin-1", error_bad_lines=False)
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageURLS', 'imageURLM', 'imageURLL']

print(books.head())

print(ratings.shape)
print(users.shape)
print(books.shape)


books.drop(['imageURLS','imageURLM','imageURLL'], axis=1, inplace=True)
print(books.head())
print(ratings.dtypes)
pd.set_option('display.max_colwidth', -1)
print(ratings['bookRating'].unique())
print(users.dtypes)
print(books.dtypes)
print(books.yearOfPublication.unique())
print(books.loc[books.yearOfPublication == 'DK Publishing Inc',:])