import pandas as pd


def readFiles():
    users = pd.read_csv('./dataSet/BX-Users.csv', delimiter=";", encoding="latin1")
    users.columns = ['userId', 'location', 'age']
    rating = pd.read_csv('./dataSet/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
    rating.columns = ['userID', 'ISBN', 'bookRating']
    print(len(users))
    print(users.head())
    print(len(rating))
    print(rating.head())

def main():
    readFiles()

if __name__ == '__main__':
    main()
