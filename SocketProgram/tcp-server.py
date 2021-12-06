import socket

PORT = 9999

# 소켓 생성 UDP , IPv4
my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 포트가 사용중이면 연결할 수 없다는 WinError 10048에러 해결 목적
my_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 바인딩
my_socket.bind(('0.0.0.0', PORT))

my_socket.listen(1)

conn, addr = my_socket.accept()

while True:
    data = conn.recv(1024)
    data = data.decode().upper()
    print('client : ', data)

    if data == 'Q':
        break

    #conn.send(data.encode())

data = conn.recv(1024)
conn.close()
my_socket.close()
