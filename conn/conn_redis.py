import redis

pool = redis.ConnectionPool(host="10.171.199.172", port=6379, password="99Apprds", db=15)
r = redis.StrictRedis(connection_pool=pool, encoding="utf-8")
keys = r.keys()

df_comment_res = pd.DataFrame()
for k in keys:
    #k = b'comment:100001107:qunar'
    if "comment:" in k.decode("utf-8"):
        val = r.get(k)
        text = val.decode("utf-8")
        text2list = eval(text)
        df_comment = pd.DataFrame.from_dict(text2list)
        #df_comment["hotelid"] = k.decode("utf-8")[8:]
        df_comment_res = pd.concat((df_comment_res, df_comment), axis=0)
    else:
        continue
