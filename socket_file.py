import socket
import struct

sample_data = [0.5,0.5,0.5]

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 11000  # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = struct.pack('f',11.1111)
            print(data)
            conn.send(data)
            if not data:
                break
