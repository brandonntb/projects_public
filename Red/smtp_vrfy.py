#!/usr/bin/python

import socket
import sys

def smtp_vrfy(username_list, target_list):
        try:
                with open(target_list, 'r') as targets:
                        targets = [target.strip() for target in targets.readlines() if target]

                
                with open(username_list, 'r') as usernames:
                        usernames = [username.strip() for username in usernames.readlines() if username]

                for target in targets:
                        print(f"\nTarget:\n --------------|{target}|--------------")
                        # Create a Socket
                        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  

                        # Connect to the Server
                        connect = s.connect((target, 25))
                        
                        # Receive the banner
                        banner =s.recv(1024)
                        
                        print(banner)
                        
                        # VRFY User
                        print("\nUsernames:")
                        for user in usernames:
                                print(f"--------------|{user}|--------------")
                                s.send(('VRFY ' + user + '\r\n').encode())
                                result = s.recv(1024)
                                print(result)

                        # Close the socket
                        s.close()
                        
        except Exception as e:
                print(e)                  

smtp_vrfy(sys.argv[1], sys.argv[2])



