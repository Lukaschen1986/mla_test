# -*- coding: utf-8 -*-
import socket

def server_func(addr_server, response):
    print("Server Start......")
    # 建立socket
    skt = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # 绑定服务端ip和port
    skt.bind(addr_server)
    # 接收内容
    data_get, addr_client = skt.recvfrom(500) # data_get: bytes
    text_get = data_get.decode("utf-8") # 解码
    print("Msg from client: " + text_get)
    # 返回信息
    print("Msg to client: " + response)
    data_post = response.encode("utf-8") # 编码
    skt.sendto(data_post, addr_client)
    print("Server End.")
    return None
    

def client_func(text_post, addr_server):
    # 建立socket
    skt = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # 编码
    print("Msg to server: " + text_post)
    data_post = text_post.encode("utf-8")
    # 发送
    skt.sendto(data_post, addr_server)
    # 接收
    data_get, addr_server = skt.recvfrom(200)
    text_get = data_get.decode("utf-8")
    print("Msg from server: " + text_get)
    return None
