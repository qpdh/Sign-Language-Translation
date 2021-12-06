import socket


class server_socket:
    # 포트번호
    PORT = 9999

    def __init__(self):
        # 소켓 생성 UDP , IPv4
        self.my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 포트가 사용중이면 연결할 수 없다는 WinError 10048에러 해결 목적
        self.my_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # 바인딩
        self.my_socket.bind(('0.0.0.0', server_socket.PORT))

        self.my_socket.listen(1)

        conn, addr = self.my_socket.accept()

        self.targetSocket = conn

        self.socketType = 0

    def close(self):
        self.targetSocket.close()
        self.my_socket.close()


class client_socket:
    # 포트번호
    PORT = 9999

    def __init__(self, ip):
        self.my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip = ip
        self.my_socket.connect((ip, client_socket.PORT))

        self.socketType = 1

    def close(self):
        self.my_socket.close()
