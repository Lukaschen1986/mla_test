# -*- coding: utf-8 -*-
from pymongo import MongoClient

conn = MongoClient(host="10.171.195.145", port=27017) # , connect=False
db = conn.sgp_hotel
my_set = db.t_hotel_smart_comment

# 增
new_data = {"hotelid": "100001308", "comment": "test_comment_1"}
my_set.insert_one(new_data)

new_data = [
        {"hotelid": "100001308", "comment": "test_comment_1"}, 
        {"hotelid": "100001308", "comment": "test_comment_2"},
        {"hotelid": "100001309", "comment": "test_comment_3"}
        ]
my_set.insert_many(new_data)

# 删
delete_data = {"hotelid": "100001308"}
my_set.delete_one(delete_data) # 删除hotelid为100001308的第一条数据
my_set.delete_many(delete_data) # 删除hotelid为100001308的所有数据
my_set.drop() # 删除集合

# 改
check_data = {"comment": "test_comment_3"}
new_value = {"$set": {"comment": "test_comment_4"}}
my_set.update_one(check_data, new_value)
my_set.update_many(check_data, new_value)


# 查
my_set.find_one()

for data in my_set.find():
    print(data)

check_data = {"comment": "test_comment_4"}
check_data = {"hotelid":100001328, "roomtypeid":100004658, "roomstatedate":{$gte:"20180701", $lte:"20180826"}}
for data in my_set.find(check_data):
    print(data)
