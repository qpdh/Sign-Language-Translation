import socket


class server_socket:
    def __init__(self):
        # 포트번호
        self.PORT = 9999

        # 소켓 생성 UDP , IPv4
        self.my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 포트가 사용중이면 연결할 수 없다는 WinError 10048에러 해결 목적
        self.my_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 바인딩
        self.my_socket.bind(('', self.PORT))

        self.socketType = 0

    # while True:
    #     data, addr = server_socket.recvfrom(1024)
    #     data = data.decode().upper()
    #     print('client : ', data)
    #     server_socket.sendto(data.encode(), addr)


class client_socket:
    def __init__(self):
        self.my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.IP = "210.99.147.179"
        # 포트번호
        self.PORT = 9999

        self.socketType = 1
    # while True:
    #     data, addr = server_socket.recvfrom(1024)
    #     data = data.decode().upper()
    #     print('client : ', data)
    #     server_socket.sendto(data.encode(), addr)
