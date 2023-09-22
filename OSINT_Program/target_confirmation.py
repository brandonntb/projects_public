import requests
from requests.exceptions import ConnectionError
from termcolor import colored, cprint


def target_conf():
    while True:
        target_domain = input('\n\t Target Domain:')
        try:
            response = requests.get(f'http://{target_domain}', timeout=10)
        except ConnectionError:
            cprint('\nINVALID TARGET', 'red', attrs=['bold'])
            print('')
            print('Would you like to try again? yes/no')
            print('*[Note]* if the answer is no, the program will attempt operations on the supplied target')

            while True:
                target_again = input('ans:').lower()
                if target_again == 'yes' or target_again == 'no':
                    break
                else:
                    print('\nPlease select a', colored('VAILD', attrs=['bold', 'underline']), 'option [yes/no]')
                    continue

            if target_again == 'yes':
                continue

            elif target_again == 'no':
                break


        else:
            cprint('\nVALID TARGET', 'green', attrs=['bold'])
            break

    return target_domain

# for debugging
#target_conf()