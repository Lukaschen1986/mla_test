# -*- coding: utf-8 -*-
from tornado import web, ioloop, httpserver
from .MySqlModel import DiaryModel

# 业务处理模块
## 首页
class MainPageHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        #self.write("Hello World.")
        self.render("index.html")
        return None

## 新建日记并入库
class CreateDiaryHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        # 新建日记页面
        self.render("create.html")
        return None

    def post(self, *args, **kwargs):
        # 接收前端post过来的信息
        weather = self.get_argument("weather")
        mood = self.get_argument("mood")
        content = self.get_argument("content")
        print(weather, mood, content)
        # 入库
        model = DiaryModel()
        diary_id = model.create_diary(weather=weather, mood=mood, content=content)
        if diary_id:
            self.write("创建成功")
        else:
            self.write("创建失败，请重新尝试")
        return None

## 查询日记列表
class DiaryListHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        # 读取数据库数据
        model = DiaryModel()
        diary_list = model.get_diary_list()
        print(diary_list)
        # 渲染到前端页面
        self.render("diary_list.html", diary_list=diary_list)
        return None

# 路由
application = web.Application([
    (r"/index", MainPageHandler),
    (r"/create", CreateDiaryHandler),
    (r"/diary_list", DiaryListHandler)
])


# socket服务器
if __name__ == "__main__":
    http_server = httpserver.HTTPServer(application)
    http_server.listen(8080)
    ioloop.IOLoop.current().start()
