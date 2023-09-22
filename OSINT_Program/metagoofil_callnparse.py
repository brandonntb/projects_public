import subprocess
from termcolor import colored
import re

metagoofil_doctypes = ['pdf', 'doc', 'xls', 'ppt', 'docx', 'pptx', 'xlsx']

metagoo_options = {'target_docs': 'pdf,doc', 'search_limit': '10', 'down_limit': '5',
                   'output_location': '../metagoofil/downloaded_files'}

# placeholder vars
usernames = []
emails = []
human_names = []
softwares = []


def metagoofil_call(target_domain):
    try:
        print('\n')
        #       here will need to make a way for user to customise options inputted

        output = subprocess.run(
            ['python', '../metagoofil/metagoofil.py', '-d', target_domain, '-t', metagoo_options['target_docs'], '-l',
             metagoo_options['search_limit'], '-n', metagoo_options['down_limit'],
             '-o', metagoo_options['output_location']], capture_output=True)

        parsed = output.stdout.decode(errors='replace').split('[+]')

#        print(parsed)

        for i in range(1, len(parsed)):
            # users parsing
            if 'users' in parsed[i][:20]:

                users = parsed[i].split()

                for n in range(5,len(users)):
                    usernames.append(users[n])

            # email parsing
            if 'e-mails' in parsed[i][:20]:

                emails_found = parsed[i].split()

                for n in range(5,len(emails_found)):
                    usernames.append(emails_found[n])

            # used software parsing
            if 'software' in parsed[i][:20]:

                soft = parsed[i].split()

                for n in range(5,len(soft)):
                    usernames.append(soft[n])

            # could also do server paths here but seems gimmicky so will leave to later as program stopped getting results

    except Exception as e:
        print('Sorry an', colored('Error', 'red'), 'has occurred with metagoofil')
        print(e, '\n')


def exif_call(target_location):
    try:
        print('\n')
        #       here will need to make a way for user to customise options inputted

        output = subprocess.run(['../exiftool/exiftool', target_location], capture_output=True)

        whats_output = output.stdout.decode(errors='replace').split('\n')

        for line in whats_output:
            if 'Author' in line[:6] or 'Creator' in line[:7]:
                if 'Creator Tool' in line:
                    pass
                else:
                    usernames.append(re.sub(r'^.*?:', '', line))

            if 'Email' in line[:5] or 'mailto' in line:
                if 'mailto' in line:
                    # subbing all the 'mailto:' found in string with nothing to delete, then using replace to
                    # delete any commas found

                    for mail in re.findall('\S+@\S+', line):
                        emails.append(re.sub(r'^.*?:', '', mail).replace(',', ''))

                else:
                    emails.append(re.sub(r'^.*?:', '', line))

            if 'Producer' in line[:8] or 'Creator Tool' in line[:12]:
                softwares.append(re.sub(r'^.*?:', '', line))


    except Exception as e:
        print('Sorry an', colored('Error', 'red'), 'has occurred with WhatsMyName')
        print(e, '\n')
