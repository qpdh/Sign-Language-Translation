import socket

IP = "1.241.73.157"
PORT = 9999

my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

my_socket.connect((IP, PORT))

while True:
    msg = input('->')
    my_socket.send(msg.encode())

    if msg == 'q':
        break
    recvMsg = my_socket.recv(1024)
    print('server : ', recvMsg.decode())

my_socket.close()
