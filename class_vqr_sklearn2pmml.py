from sklearn_pandas import DataFrameMapper
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn.preprocessing import LabelBinarizer # str to one-hot
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder # int to one-hot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import FunctionTransformer 

columns = DataFrameMapper(
        features=[
                (["orderattribute"], FunctionTransformer(np.int32)),
                (["ordersource"], [FunctionTransformer(np.int32), OneHotEncoder()]),
                (["payway"], [FunctionTransformer(np.int32), OneHotEncoder()]),
                (["cityid"], LabelBinarizer()),
                (["hotelstargrade"], [FunctionTransformer(np.int32), OneHotEncoder()]),
                (["contractprice"], FunctionTransformer(np.int32)),
                (["contractprice_disperse"], LabelBinarizer()),
                (["guaranstatus"], FunctionTransformer(np.int32)),
                (["create_late"], FunctionTransformer(np.int32)),
                (["create_sameday"], FunctionTransformer(np.int32)),
                (["create_twoday"], FunctionTransformer(np.int32)),
                (["pre_hours"], FunctionTransformer(np.float32)),
                (["pre_hours_disperse"], LabelBinarizer()),
                (["room_nights"], FunctionTransformer(np.int32)),
                (["room_nights_disperse"], LabelBinarizer()),
                (["create_month"], LabelBinarizer()),
                (["create_day"], [FunctionTransformer(np.int32), OneHotEncoder()]),
                (["create_hour"], LabelBinarizer()),
                (["create_weekday"], LabelBinarizer()),
                (["checkin_month"], LabelBinarizer()),
                (["checkin_day"], [FunctionTransformer(np.int32), OneHotEncoder()]),
                (["checkin_weekday"], LabelBinarizer())
                ],
        default=False
        )
       
pipeline = PMMLPipeline([
        ("columns", columns),
        ("XGBClassifier", clf)
])
pipeline.fit(x, y)
sklearn2pmml(pipeline, 
             pmml="./XGBClassifier.pmml",
             with_repr=True, # 是否打印模型参数到pmml
             debug=False)
