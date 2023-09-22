from art import *
from file_analyzers import *
from username_searchers import *
from dns_enumerators import *
from assorted_funcs import *

from timeit import default_timer as timer

remote_global_choice = ''


def Banner():
    print('*' * 100)
    tprint('Final')
    tprint('Wrapper')
    tprint('Program')
    print('*' * 100)
    print('\n')
    print('-' * 100)
    print('This program was coded by', colored('Brandon Thomason Boyd', 'cyan', attrs=['bold']),
          'for the purpose of a final year dissertation')
    print('-' * 100)

    print('\n')


def main_options():
    print('\nWelcome to the final OSINT solution, please follow the next steps:')
    print('---------------------------------------------------------------')
    print('First, please select an OSINT task:\n')
    print('[-] 1. Pure DNS enumeration / subdomain discovery')
    print('[-] 2. File download and metadata analysis based on a domain target')
    print('[-] 3. Username checking (validating if a username can be found across multiple sources)')
    print('[-] 4. Close program')


def Menu():
    main_options()

    while True:
        mAns = str(input())

        # DNS ----------------------------------------------------------------------------------------------------
        if mAns == '1':
            print('')
            cprint('[+] 1. Pure DNS enumeration / subdomain discovery', 'green', attrs=['underline', 'bold'])
            print('What is the target domain? Please enter below:\n')

            target_domain = input('ans:').strip()

            # target_domain = 'coventry.ac.uk'

            the_harvester_callnparse(target_domain)

            main_options()

        # File Analysis ------------------------------------------------------------------------------------------------
        elif mAns == '2':
            global remote_global_choice
            print('')
            cprint('[+] 2. File download and metadata analysis based on a domain target', 'green',
                   attrs=['underline', 'bold'])
            print('What is the target domain? Please enter below:')

            target_domain = input('ans:').strip()

            # target_domain = 'coventry.ac.uk'

            print("Please type '1' if you would like to perform local and remote analysis independently")
            remote_local_choice = input('ans:').strip()

            if remote_local_choice == '1':
                print("\nPlease supply a filepath for the directory to locally analyse")
                target_location = input('ans:').strip()

            else:
                print('[*]', colored('Running sequentially', 'red'))
                target_location = '../metagoofil/downloaded_files'

            start_file = timer()

            if multi_file(target_domain, target_location, remote_global_choice) is not False:
                metagoofil_parse()
                exif_parse()

                if temp_file.tell() == 0:
                    cprint('\nUsers:', 'yellow', attrs=['bold', 'underline'])
                    orderuniq(usernames)

                    cprint('\nEmails:', 'yellow', attrs=['bold', 'underline'])
                    orderuniq(emails)

                    cprint('\nSoftware:', 'yellow', attrs=['bold', 'underline'])
                    orderuniq(software)

            temp_file.close()

            end_file = timer()

            print(f'\nFile analysis completed operations in {end_file - start_file:0.4f} seconds')

            print(
                "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

            return False

        # Username checking --------------------------------------------------------------------------------------------
        elif mAns == '3':
            print('')
            cprint('[+] 3. Username checking (validating if a username can be found across multiple sources)', 'green',
                   attrs=['underline', 'bold'])
            print('What is the target username? Please enter below:')

            #   Getting username
            target_username = input('\n\t Target Username:')

            # target_username = 'Cielilous Ossint'
            print('')

            if ' ' in target_username:
                print("Usernames dont have spaces, all spaces have been removed from the target username, if you made "
                      "a mistake please try again.\n")

                remspace = target_username.replace(" ", "")

            else:
                remspace = target_username

            start_username = timer()

            if multi_username(remspace) is not False:
                #  SHERLOCK
                sherlock_parse()
                #  WHATS MY NAME
                whatsMyName_parse()

                if temp_file.tell() == 0:
                    cprint('\nPlatforms:', 'yellow', attrs=['bold', 'underline'])
                    orderuniq(valid_platforms)

                    cprint('\nValid URLs:', 'yellow', attrs=['bold', 'underline'])
                    orderuniq(valid_urls)

            temp_file.close()
            end_username = timer()

            print(f'\nUsername checking completed operations in {end_username - start_username:0.4f} seconds')

            print(
                "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")

            return False

        elif mAns == '4':
            temp_file.close()
            return False

        else:
            print('\nPlease select a valid option [1-4]')


Banner()
Menu()
