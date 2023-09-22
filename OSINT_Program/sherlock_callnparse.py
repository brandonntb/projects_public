import subprocess

def sherlock_call(target_username):
    sher_output = subprocess.run(['python3', '../sherlock/sherlock/sherlock.py', target_username], capture_output=True)



def whatsMyName_call(target_username):
    whatsName_output = subprocess.run(['python3', '../WhatsMyName/web_accounts_list_checker.py', '-u',target_username], capture_output=True)