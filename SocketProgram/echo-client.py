import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

IP = "127.0.0.1"

while True:
    msg = input('->')
    client_socket.sendto(msg.encode(), (IP, 9999))
    recvMsg, addr = client_socket.recvfrom(2048)
    print('server : ', recvMsg.decode())
