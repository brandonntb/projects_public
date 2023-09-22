from assorted_funcs import *
import re
import subprocess
import tempfile

metagoofil_doctypes = ['pdf', 'doc', 'xls', 'ppt', 'docx', 'pptx', 'xlsx']

metagoo_options = {'target_docs': 'pdf,doc', 'search_limit': '10', 'down_limit': '5',
                   'output_location': '../metagoofil/downloaded_files'}

# placeholder vars
usernames = []
emails = []
software = []

m_out = ''
m_err = ''

e_out = ''
e_err = ''

verbose_choice = ''




def multi_file(target_domain, target_location, choice):
    global m_out, m_err, e_out, e_err, verbose_choice

    verbose_choice = verbose()

    print(colored('\nMetagoofil', 'cyan', attrs=['bold']), 'executing ...\n')

    # USER CUSTOM OPTIONS DISABLED FOR TESTING ---------------------------------------------------------

    print('What document types should the program scan for? (example input: pdf,doc,xls)\n')
    print('Doc Type options are as follows:')
    for i in metagoofil_doctypes:
        print(i)
    metagoo_options['target_docs'] = input('\nans:').lower().strip()

    for i in metagoo_options['target_docs'].split(','):
        if i.strip() in metagoofil_doctypes:
            pass

        else:
            print(colored(i, 'red', attrs=['bold']), '-> is not a valid choice, please try again with only valid options included')

            return False

    print('\nHow many results should the program search?')
    metagoo_options['search_limit'] = input('ans:').strip()

    print('\nHow many files should the program download for analysis?')
    metagoo_options['down_limit'] = input('ans:').strip()

    if choice == 1:
        p_meta = subprocess.Popen(
            ['python', '../metagoofil/metagoofil.py', '-d', target_domain, '-t', metagoo_options['target_docs'], '-l',
             metagoo_options['search_limit'], '-n', metagoo_options['down_limit'],
             '-o', metagoo_options['output_location']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(colored('\nExiftool', 'cyan', attrs=['bold']), 'executing ...\n')
        p_exif = subprocess.Popen(['../exiftool/exiftool', target_location], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

        m_out, m_err = p_meta.communicate()
        e_out, e_err = p_exif.communicate()

    else:
        p_meta = subprocess.Popen(
            ['python', '../metagoofil/metagoofil.py', '-d', target_domain, '-t', metagoo_options['target_docs'], '-l',
             metagoo_options['search_limit'], '-n', metagoo_options['down_limit'],
             '-o', metagoo_options['output_location']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        m_out, m_err = p_meta.communicate()

        print(colored('\nExiftool', 'cyan', attrs=['bold']), 'executing ...\n')
        p_exif = subprocess.Popen(['../exiftool/exiftool', target_location], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)

        e_out, e_err = p_exif.communicate()




def metagoofil_parse():
    print(colored('\nMetagoofil', 'cyan', attrs=['bold']), 'parsing ...\n')
    try:

        decoded = m_out.decode('utf-8', errors='ignore').split('[+]')

        if verbose_choice == 'no':
            for i in range(1, len(decoded)):
                # users parsing
                if 'users' in decoded[i][:20]:

                    users = decoded[i].split()

                    for n in range(5, len(users)):
                        usernames.append(users[n])

                # email parsing
                elif 'e-mails' in decoded[i][:20]:

                    emails_found = decoded[i].split()

                    for n in range(5, len(emails_found)):
                        emails.append(emails_found[n])

                # used software parsing
                elif 'software' in decoded[i][:20]:

                    soft = decoded[i].split()

                    for n in range(5, len(soft)):
                        software.append(soft[n])

            if len(usernames) < 1:
                cprint('No Metagoofil results, please try again', 'red')

        else:
            for i in decoded:
                temp_file.write(i + '\n')

            temp_file.seek(0)
            print(temp_file.read())

    except Exception as e:
        print('[*] Sorry an', colored('Error', 'red'), 'has occurred with the Metagoofil parser:')
        print(e, '\n')


def exif_parse():
    print(colored('\nExifTool', 'cyan', attrs=['bold']), 'parsing ...\n')
    try:
        decoded = e_out.decode('utf-8', errors='ignore').split('\n')

        if verbose_choice == 'no':
            for line in decoded:
                if 'Author' in line[:6] or 'Creator' in line[:7]:
                    if 'Creator Tool' in line:
                        pass
                    else:
                        usernames.append(re.sub(r'^.*?:', '', line))

                elif 'Email' in line[:5] or 'mailto' in line:
                    if 'mailto' in line:
                        # subbing all the 'mailto:' found in string with nothing to delete,
                        # then using replace to
                        # delete any commas found

                        # get find email address in the line
                        for mail in re.findall('\S+@\S+', line):
                            emails.append(re.sub(r'^.*?:', '', mail).replace(',', ''))

                    else:
                        emails.append(re.sub(r'^.*?:', '', line))

                elif 'Producer' in line[:8] or 'Creator Tool' in line[:12]:
                    software.append(re.sub(r'^.*?:', '', line))

        else:
            for i in decoded:
                temp_file.write(i + '\n')

            temp_file.seek(0)
            print(temp_file.read())

    except Exception as e:
        print('[*] Sorry an', colored('Error', 'red'), 'has occurred with the ExifTool parser:')
        print(e, '\n')
