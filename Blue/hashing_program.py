import math
import time
import datetime

password = "EscapeFromTarkov"

#stores password sting in list for each character i.e. b,r,a,n,d,o,n
password = list(password)

hash_pass = []
hashed_password = ''

counter = 33 

currentDate = datetime.datetime.now()

def convert_to_ascii(passw):
    for i in range(0,len(passw)):
        passw[i] = ord(passw[i]) # convertin each char to its ascii equivelent

def ascii_range(n):
    global counter # making global so it will change the variable outside the functs scope
                    #keeping within ascii value limits
    for x in range(0, int(n)):
        counter += 1   # this ensures the counter wont be reset, thus making it harder 
        #print(counter) # to ascertain the process used on the hash
        if counter > 64 and counter < 91: #this just skips the uppercase letters in the ascii table
            counter = 91
        if counter > 126:
            counter = 33

def salt(passw): #salt could be list of all running processes - ps util in python 
    temp_counter = 0
    creationDate = currentDate.strftime("%Y/%m/%d").split('/')
    creationTime = currentDate.strftime("%S:%M:%H").split(':')
    
    for c in creationTime: #inserting time salt in steps of 2
        c = int(c)
        passw.insert(temp_counter, c)
        temp_counter += 2
        
    for x in creationDate: #inserting date salt in steps of 2
        x = int(x)
        passw.insert(temp_counter, x)
        temp_counter +=2


def pad(passw): #pads the pasword out to 24 characters each time irregardles of lenght, the salt is the disperesed throught the whole string
    pad = ["%","e","^","h","?","<","*","g","@","q","|"]
    if len(passw) < 60: #could pad to 60, or 40 or cut off pass if over 30 chars long unsure which to do 
        for i in range(0, 60 - len(passw)):
            passw.append(ord(pad[len(passw)%len(pad)])) # the modulo here prevents esscaping the index range the pad
            #print("TESTING -> pad added: ", passw) 

"""
    if len(passw) > 30:
        for i in range(0, len(passw), 3):
            if i > len(passw):
                i = i - len(passw)
                passw.remove(passw[i])
"""         
    

"""
passw.append(ord(pad[i]))

passw.insert(0, ord(pad2[i]))
"""

def transform(passw):
    key = 122753
    for i in range(3,len(passw) + 3):
        #print("\n--> Ascii values before operation (%s out of %s): " %(i - 2,len(passw)), passw[i - 3])

        # math operations to change values
        newVal = passw[i-3] + ( (passw[i - 3] * (key * i))  /  (key/14) )
        
        ascii_range(newVal)
              
        #setting new value within ascii limits
        passw[i - 3] = counter


        #consider transposing these in a differnt order
        #hash_pass.insert(0,chr(passw[i - 3])) # chr converts ascii back to char
        #insert is pushin them back in backwards so pass string is reversed

        hash_pass.append(chr(passw[i - 3]))

def checkforduplicates(passw):
    for i in range(0, len(passw) - 1):
        if(passw[i] == passw[i+1]):
            print("\n\n-------------->DUPLICATE DETECTED at index ",i," and ", i+1,"<------------------")
            print("\n\t\tThese elements are ->", passw[i], " and ->", passw[i+1])
        

    
print("Password before hash:" , password)

convert_to_ascii(password)
print("Changed to ascii: ", password, "\n")


salt(password)
print("Added the salt: ", password)

pad(password)
print("\nPadded password: ", password)


transform(password)
print("\nAscii values after all operations: ", password)


print("")

print("Hashed password: ", hash_pass)

checkforduplicates(hash_pass)

fake_count = 0
for x in hash_pass:
    hashed_password += x
    fake_count +=1
print(fake_count) # counting how many charas are in the  hashed password
print("\nHashed password string: ", hashed_password)


