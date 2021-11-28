import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

IP = "210.99.147.179"

while True:
    msg = input('->')
    client_socket.sendto(msg.encode(), (IP, 9999))
    recvMsg, addr = client_socket.recvfrom(8000)
    print('server : ', recvMsg.decode())
