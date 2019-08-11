import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import numpy as np

bookName = "The Green Mile: Coffey's Hands (Green Mile Series)"

def knn_main(bookName):
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
    user_rating_matrix = csr_matrix(user_rating_pivot.values)

    # build knn model
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(user_rating_matrix)
    result = NearestNeighbors(algorithm='brute', leaf_size=30, metric='cosine', metric_params=None, n_jobs=1, n_neighbors=5, p=2, radius=1.0)


    # generate a random index of book for making prediction
    query_index = np.random.choice(user_rating_pivot.shape[0])

    # bookName = "The Green Mile: Coffey's Hands (Green Mile Series)"
    for i in range(2442):
        if (user_rating_pivot.index[i] == bookName):
            # print(i)
            query_index = i
            break
    # print(user_rating_pivot.index[:10])

    distances, indices = model_knn.kneighbors(user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 10)

    # get result
    result =[]
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('\nKNN recommendations for {0}:\n'.format(user_rating_pivot.index[query_index]))
        else:
            recommended = user_rating_pivot.index[indices.flatten()[i]]
            result.append(recommended)
            print("%d: %s, with distance of %.2f:" %(i,recommended,distances.flatten()[i]))
    print()
    return result

if __name__ == '__main__':
    result = knn_main(bookName)