import subprocess
import re
import tempfile
from assorted_funcs import *

#       SHERLOCK CAN SEARCH FOR MORE THAN ONE USER, CREATE A WAY FOR THE USER TO DO THIS

actual_names = []
maybe_names = []

platforms_found = []
urls_found = []

valid_platforms = []
valid_urls = []

s_out = ''
s_err = ''

w_out = ''
w_err = ''

verbose_choice = ''


def multi_username(target_username):
    global s_out, s_err, w_out, w_err, verbose_choice

    verbose_choice = verbose()

    print(colored('\nSherlock', 'cyan', attrs=['bold']), 'executing ...\n')
    p_sher = subprocess.Popen(['python3', '../sherlock/sherlock/sherlock.py', target_username], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    print(colored('\nWhatsMyName', 'cyan', attrs=['bold']), 'executing ...\n')
    p_what = subprocess.Popen(['python3', '../WhatsMyName/web_accounts_list_checker.py', '-u', target_username],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    s_out, s_err = p_sher.communicate()
    w_out, w_err = p_what.communicate()

    p_sher.terminate()
    p_what.terminate()


def sherlock_parse():
    print(colored('\nSherlock', 'cyan', attrs=['bold']), 'parsing ...\n')
    try:
        decoded = s_out.decode('utf-8', errors='ignore').split('\n')
        #    print(decoded)

        if verbose_choice == 'no':
            for line in decoded:
                # print('Line:', line)
                platforms = re.search(r'] (.*?): ', line)
                if platforms is not None:
                    platforms_found.append(platforms.group(1))

                urls = re.search(r'(http[s]?://[^\s]+)', line)
                if urls is not None:
                    urls_found.append(urls.group(1))

            for i in range(0, len(urls_found)):
                if testurl(urls_found[i]):
                    valid_urls.append(urls_found[i])
                    valid_platforms.append(platforms_found[i])

        else:

            for i in decoded:
                temp_file.write(i + '\n')

            temp_file.seek(0)
            print(temp_file.read())


    except Exception as e:
        print('[*] Sorry an', colored('Error', 'red'), 'has occurred with the Sherlock parser:')
        print(e, '\n')


def whatsMyName_parse():
    print(colored('\nWhatsMyName', 'cyan', attrs=['bold']), 'parsing ...\n')
    try:
        decoded = w_out.decode('utf-8', errors='ignore').split('\n')

        if verbose_choice == 'no':
            for line in decoded:
                # print('Line:', line)
                userURLs = re.search(r'Found user at (.*?)\x1b\[0m', line)

                if userURLs is not None:
                    if testurl(userURLs.group(1)) or ".api" in userURLs.group(1):
                        valid_urls.append(userURLs.group(1))

        else:

            for i in decoded:
                temp_file.write(i + '\n')

            temp_file.seek(0)
            print(temp_file.read())

    except Exception as e:
        print('[*] Sorry an', colored('Error', 'red'), 'has occurred with the WhatsMyName parser:')
        print(e, '\n')
