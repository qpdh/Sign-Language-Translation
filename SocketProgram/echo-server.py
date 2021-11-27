import socket

# 포트번호
PORT = 9999

# 소켓 생성 UDP , IPv4
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 포트가 사용중이면 연결할 수 없다는 WinError 10048에러 해결 목적
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 바인딩
server_socket.bind(('', PORT))

while True:
    data, addr = server_socket.recvfrom(1024)
    data = data.decode().upper()
    print('client : ', data)
    server_socket.sendto(data.encode(), addr)
