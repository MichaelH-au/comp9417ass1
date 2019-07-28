
combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
print(f"\ncombine_book_rating:\n{combine_book_rating.head()}")

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])
book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )
print(f"\nbook_ratingCount:\n{book_ratingCount.head()}")

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')
print(f"\nrating_with_totalRatingCount:\n{rating_with_totalRatingCount.head()}")

pd.set_option('display.float_format', lambda x: '%.3f' % x)
print("book_ratingCount:")
print(book_ratingCount['totalRatingCount'].describe())
