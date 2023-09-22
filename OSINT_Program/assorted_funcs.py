import requests
from requests.exceptions import ConnectionError
from termcolor import colored, cprint
import tempfile

temp_file = tempfile.TemporaryFile(mode='w+')

def target_conf():
    while True:
        target_domain = input('\n\t Target Domain:')
        try:
            response = requests.get(f'http://{target_domain}', timeout=10)
        except ConnectionError:
            cprint('\nINVALID TARGET', 'red', attrs=['bold'])
            print('')
            print('Would you like to try again? yes/no')
            print('[*] if the answer is no, the program will attempt operations on the supplied target')

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

        cprint('\nVALID TARGET', 'green', attrs=['bold'])

    return target_domain


def testurl(target_url):
    try:
        request = requests.get(target_url)

        # print(request.status_code)
        if request.status_code < 400 or request.status_code >= 500:
            if request.status_code < 400:
                return True

            else:
                print("[*] The URL:", colored(target_url, 'red', attrs=['bold']), "seems to be having server issues but "
                                                                              "is still a valid URL please try again "
                                                                              "as this will likely be fixed \n")
                return True

        else:
            return False

    except Exception as e:
        print('[*] Sorry an', colored('Error', 'red'), 'has occurred within the URL testing:')
        print(e, '\n')


def orderuniq(list):
    #           sorted sorts the list a-z and set makes sure only uniq results are printed
    if len(list) > 0:
        for i in range(0, len(sorted(set(list)))):
            if list[i]:
                print(f'{i + 1}. {list[i]}')

    else:
        cprint('No data found', 'red')


def verbose():
    print('\nWould you like to see the results in verbose mode? (this will display the raw output '
          'from the programs, useful for error checking) [yes/no]\n')

    while True:
        verbose_choice = input('ans:').lower()
        # verbose_choice = 'no'
        if verbose_choice == 'yes' or verbose_choice == 'no':
            break
        else:
            print('Please select a valid option (yes/no)')

    return verbose_choice

