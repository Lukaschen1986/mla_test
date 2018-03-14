import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.grid_search import GridSearchCV 
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer 
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import FunctionTransformer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import scorer

testdata = pd.DataFrame({'pet':['cat', 'dog', 'dog', 'fish', 'cat', 'dog', 'cat', 'fish'],
                         'age': [4, 6, 3, 3, 2, 3, 5, 4],
                         'salary': [90, 24, 44, 27, 32, 59, 36, 27]})

mapper = DataFrameMapper(
        features=[
        (["age","salary"], MinMaxScaler()),
        ("pet", LabelBinarizer())
        ], 
        default=None
        )

mapper = DataFrameMapper(
        features=[        
        (["age","salary"], FunctionTransformer(np.log1p))    
        ],
        default=False
        ) 

# default=False 全部丢弃（默认）
# default=None 原封不动地保留
# np.log1p log(x+1)
