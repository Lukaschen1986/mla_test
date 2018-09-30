# -*- coding: utf-8 -*-
import socket

def server_func(addr_server, response):
    print("Server Start......")
    # 建立socket
    skt_server = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    # 绑定服务端ip和port
    skt_server.bind(addr_server)
    # 监听接入的socket
    skt_server.listen()
    # 死循环
    while True:
        # 接受访问
        skt_client, addr_client = skt_server.accept()
        # 接收内容
        data_get = skt_client.recv(500)
        text_get = data_get.decode("utf-8") # 解码
        print("Msg from client: " + text_get)
        # 数据计算与交互
        '''
        algorithm
        '''
        # 返回信息
        print("Msg to client: " + response)
        data_post = response.encode("utf-8") # 编码
        #skt_client.sendto(data_post, addr_client)
        skt_client.send(data_post)
        # 关闭连接通路
        skt_client.close()
    return None


def client_func(text_post, addr_server):
    # 建立socket
    skt_client = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    # 建立连接通路
    skt_client.connect(addr_server)
    # 编码
    print("Msg to server: " + text_post)
    data_post = text_post.encode("utf-8")
    # 发送
    skt_client.send(data_post)
    # 接收
    data_get = skt_client.recv(200)
    text_get = data_get.decode("utf-8")
    print("Msg from server: " + text_get)
    # 关闭连接通路
    skt_client.close()
    return None
