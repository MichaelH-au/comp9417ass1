import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

print("\nProgram starts.\n")

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

print(f"\nratings.shape:\n{ratings.shape}")
print(f"\nratings.columns:\n{list(ratings.columns)}")

# # Rating distribution
# plt.rc("font", size=15)
# ratings.bookRating.value_counts(sort=False).plot(kind='bar')
# plt.title('Rating Distribution\n')
# plt.xlabel('Rating')
# plt.ylabel('Count')
# plt.savefig('system1.png', bbox_inches='tight')
# plt.show()

# # Age distribution
# users.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
# plt.title('Age Distribution\n')
# plt.xlabel('Age')
# plt.ylabel('Count')
# plt.savefig('system2.png', bbox_inches='tight')
# plt.show()

# Recommendaion based on ratings
print("\nRecommendaion based on ratings")
rating_count = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
print(rating_count.sort_values('bookRating', ascending=False).head())

most_rated_books = pd.DataFrame(['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
print(f"\nmost_rated_books_summary:\n{most_rated_books_summary}")

# Recommendation based on relevance
print("\nRecommendation based on relevance")
average_rating = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].mean())
average_rating['ratingCount'] = pd.DataFrame(ratings.groupby('ISBN')['bookRating'].count())
print(average_rating.sort_values('ratingCount', ascending=False).head(10))

# Remove users with less than 200 ratings and books with less than 100 ratings
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

# Rating Matrix
ratings_pivot = ratings.pivot(index='userID', columns='ISBN').bookRating
userID = ratings_pivot.index
ISBN = ratings_pivot.columns
print(ratings_pivot.shape)
print(ratings_pivot.head())

print("\nProgram complete.")