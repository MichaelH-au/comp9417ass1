import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

book_name = "The Green Mile: Coffey's Hands (Green Mile Series)"

def matrix_fact(book_name):
    # initialize all the data models
    books = pd.read_csv('Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher']
    users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ['userID', 'Location', 'Age']
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    ratings.columns = ['userID', 'ISBN', 'bookRating']

    # make combination of tables
    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

    # count the number of rating for each book
    book_ratingCount = (combine_book_rating.
        groupby(by = ['bookTitle'])['bookRating'].
        count().
        reset_index().
        rename(columns = {'bookRating': 'totalRatingCount'})
        [['bookTitle', 'totalRatingCount']]
        )

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')

    # ignore the books that rating less than 50
    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

    combined = rating_popular_book.merge(users, left_on = 'userID', right_on = 'userID', how = 'left')

    user_rating=combined.drop('Age', axis=1)

    user_rating_pivot = user_rating.pivot_table(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
    user_rating_pivot_matrix = csr_matrix(user_rating_pivot.values)

    user_rating_pivot2 = user_rating.pivot_table(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)
    # print(user_rating_pivot2.head())


    # user_rating_pivot2.shape
    # (40017, 2442)

    X = user_rating_pivot2.values.T
    # X.shape

    import sklearn
    from sklearn.decomposition import TruncatedSVD

    SVD = TruncatedSVD(n_components=12, random_state=17)
    matrix = SVD.fit_transform(X)
    # matrix.shape

    import warnings
    warnings.filterwarnings("ignore",category =RuntimeWarning)
    corr = np.corrcoef(matrix)
    # corr.shape

    us_canada_book_title = user_rating_pivot2.columns
    us_canada_book_list = list(us_canada_book_title)
    coffey_hands = us_canada_book_list.index(book_name)
    # print("\ncoffey_hands =",coffey_hands)

    corr_coffey_hands  = corr[coffey_hands]
    # print("\ncorr_coffey_hands =",corr_coffey_hands)
    # print("\nus_canada_book_title =",us_canada_book_title)

    result = list(us_canada_book_title[(corr_coffey_hands >= 0.9)])
    print(f"\nMatrix factorization recommendation for {book_name}:\n")
    for e in result:
        print(e)
    print("\nMatrix factorization Complete.")
    return result


if __name__ == '__main__':
    matrix_fact(book_name)