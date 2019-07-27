import pandas as pd
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objects as go
import plotly.offline as pltoff
from surprise import *
from surprise.model_selection import cross_validate
from surprise.accuracy import rmse
from surprise.model_selection import train_test_split
from surprise import accuracy





users = pd.read_csv('./dataSet/BX-Users.csv', delimiter=";", encoding="latin1")
users.columns = ['userID', 'location', 'age']
rating = pd.read_csv('./dataSet/BX-Book-Ratings.csv', delimiter=";", encoding="latin1")
rating.columns = ['userID', 'ISBN', 'bookRating']

def readFiles():
    print("length of users: " + str(len(users)))
    # print(users.head())
    print("length of ratings: " + str(len(rating)))
    # print(rating.head())

def mergeUserAndRatings():
    df = pd.merge(users, rating, on='userID', how='inner')
    # df.drop(['location', 'age'], axis=1, inplace=True)
    # print(len(df))
    # print(df.head())
    # DisOfRatings(df)
    # disOfBooks(df)
    topTenBooks = df.groupby('ISBN')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
    # print(topTenBooks)
    # disRatingAndUsers(df)
    topUserRating = df.groupby('userID')['bookRating'].count().reset_index().sort_values('bookRating', ascending=False)[:10]
    print(topUserRating)
    filter(df)

def DisOfRatings(df):
    data = df['bookRating'].value_counts().sort_index(ascending=False)
    trace = go.Bar(x=data.index,
                   text=['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
                   textposition='auto',
                   textfont=dict(color='#000000'),
                   y=data.values,)
    layout = dict(title='Distribution Of {} book-ratings'.format(df.shape[0]),
                  xaxis=dict(title='Rating'),
                  yaxis=dict(title='Count'))
    print(layout)
    fig = go.Figure(data=[trace], layout=layout)
    pltoff.plot(fig, filename="test")

def disOfBooks(df):
    data = df.groupby('ISBN')['bookRating'].count().clip(upper=50)

    # Create trace
    trace = go.Histogram(x=data.values,
                         name='Ratings',
                         xbins=dict(start=0,
                                    end=80,
                                    size=2))
    # Create layout
    layout = go.Layout(title='Distribution Of Number of Ratings for Books (Clipped at 50)',
                       xaxis=dict(title='Number of Ratings for Books'),
                       yaxis=dict(title='Count'),
                       bargap=0.2)
    fig = go.Figure(data=[trace], layout=layout)
    pltoff.plot(fig, filename="test")

def disRatingAndUsers(df):
    data = df.groupby('userID')['bookRating'].count().clip(upper=50)

    # Create trace
    trace = go.Histogram(x=data.values,
                         name='Ratings',
                         xbins=dict(start=0,
                                    end=50,
                                    size=2))
    # Create layout
    layout = go.Layout(title='Distribution Of Number of Ratings for Users (Clipped at 50)',
                       xaxis=dict(title='Number of Ratings for Users'),
                       yaxis=dict(title='Count'),)
    fig = go.Figure(data=[trace], layout=layout)
    pltoff.plot(fig, filename="test")

def filter(df):
    min_book_ratings = 50
    filter_books = df['ISBN'].value_counts() > min_book_ratings
    filter_books = filter_books[filter_books].index.tolist()

    min_user_ratings = 50
    filter_users = df['userID'].value_counts() > min_user_ratings
    filter_users = filter_users[filter_users].index.tolist()

    df_new = df[(df['ISBN'].isin(filter_books)) & (df['userID'].isin(filter_users))]
    print('The original data frame shape:\t{}'.format(df.shape))
    print('The new data frame shape:\t{}'.format(df_new.shape))
    # findBestAlgorithm(df_new)
    train(df_new)

def findBestAlgorithm(data):
    benchmark = []
    # 尝试所有算法
    print('---------start---------')
    reader = Reader(rating_scale=(0, 9))
    print(reader)
    data = Dataset.load_from_df(data[['userID', 'ISBN', 'bookRating']], reader)
    for algorithm in [SVD(),  SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(),
                      KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # 在交叉验证集上的表现
        results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        print("--")
        print(results)
        # 记录结果
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)

    print(benchmark)
    surprise_results = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
    print(surprise_results)

def train(data):

    reader = Reader(rating_scale=(0, 9))
    data = Dataset.load_from_df(data[['userID', 'ISBN', 'bookRating']], reader)
    print('Using ALS')
    bsl_options = {'method': 'als',
                   'n_epochs': 5,
                   'reg_u': 12,
                   'reg_i': 5
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    print(cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False))
    trainset, testset = train_test_split(data, test_size=0.25)
    algo = BaselineOnly(bsl_options=bsl_options)
    predictions = algo.fit(trainset).test(testset)
    print(accuracy.rmse(predictions))

    def get_Iu(uid):
        """ return the number of items rated by given user
        args:
          uid: the id of the user
        returns:
          the number of items rated by the user
        """
        try:
            return len(trainset.ur[trainset.to_inner_uid(uid)])
        except ValueError:  # user was not part of the trainset
            return 0


    def get_Ui(iid):
        """ return number of users that have rated given item
        args:
          iid: the raw id of the item
        returns:
          the number of users that have rated the item.
        """
        try:
            return len(trainset.ir[trainset.to_inner_iid(iid)])
        except ValueError:
            return 0
    df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
    df['Iu'] = df.uid.apply(get_Iu)
    df['Ui'] = df.iid.apply(get_Ui)
    df['err'] = abs(df.est - df.rui)

    best_predictions = df.sort_values(by='est')[:10]
    worst_predictions = df.sort_values(by='err')[-10:]
    print(best_predictions)
    print(worst_predictions)
    # best_predictions.to_csv("result.csv")




def main():
    readFiles()
    mergeUserAndRatings()


if __name__ == '__main__':
    main()
