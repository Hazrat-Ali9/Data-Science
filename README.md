# Hazrat Ali

# Software Engineering




# from matplotlib import style
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# from matplotlib.gridspec import GridSpec
# from plotly.subplots import make_subplots
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)
# sns.set()
# #style.use('fivethirtyeight')
# pd.options.mode.chained_assignment = None


# Data Science Math Command : 
 
    df = pd.read_csv('dataset.csv')
    df.head()
    df.tail()
    df.info()
    df.describe()
    df.isnull().sum()
    df.columns
    df.hist(figsize=(20,15))
    plt.show()
    sns.pairplot(df.iloc[:,0:6])
    plt.show()
    df.drop(['name'],axis=1,inplace=True)
    df.corr()["status"][:-1].sort_values().plot(kind="bar")



    

