import socket

IP = "210.99.147.179"
PORT = 9999

my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

my_socket.connect((IP, PORT))

while True:
    msg = input('->')
    my_socket.send(msg.encode())
    recvMsg, addr = my_socket.recv(1024)
    print('server : ', recvMsg.decode())
