import subprocess
from assorted_funcs import *
import re

harvester_sources = ['all', 'baidu', 'bing', 'bingapi', 'dogpile', 'google',
                     'googleCSE', 'googleplus', 'google-profiles', 'linkedin', 'pgp', 'twitter',
                     'vhost', 'virustotal', 'threatcrowd', 'crtsh', 'netcraft', 'yahoo']

harvester_options = {'data_source': '-b', 'start_at_result_number X': '-S', 'dns_resolution_n_virtual_hosts': '-v',
                     'output_results': '-f', 'dns_lookup': '-n', 'brute_dns': '-c', 'dns_TLD_expansion': '-t',
                     'use_specific_dns_server': '-e', 'use_proxies': '-p', 'search_limit': '-l',
                     'query_shodan_w_results': '-s', 'check_takeover': '-r', 'use_googleDork': '-g'}


Users = []
Emails = []
URLs = []
Hosts = []
Ips = []


def the_harvester_callnparse(target_domain):
    print(colored('\nTheHarvester', 'cyan', attrs=['bold']), 'executing ...\n')
    try:
        verbose_choice = verbose()

        print('\nWhat sources should the program use? (example input: google,bing)\n')
        print('The source options are as follows (sources are case sensitive):')
        for i in range(0, len(harvester_sources)):
            print(str(i + 1) + '.', harvester_sources[i])

        source_choice = input('\nans:').strip()

        for i in source_choice.split(','):
            if i.strip() in harvester_sources:
                pass

            else:
                print(colored(i, 'red',attrs=['bold']), colored('is not a valid choice, please try again using only '
                                                                'valid options', 'red'))

                return False

        print('\nWould you like to enable brute forcing? [yes/no]')
        brute = input('ans:').lower().strip()

        print('\nWould you like to enable DNS reverse lookups? [yes/no]')
        reverseDNS = input('ans:').lower().strip()

        print('\nHow many searches should the program perform? (please provide a number)')
        search_limit = input('ans:').strip()


        if brute == 'yes' and reverseDNS != 'yes':
            print('\n[*]', colored('Running the program with brute forcing and without reverse lookups ...', 'red'))
            harvester_output = subprocess.run(['python3', '../theHarvester/theHarvester.py', '-d', target_domain,harvester_options['search_limit'],search_limit,
                                               harvester_options['data_source'], source_choice,
                                               harvester_options['brute_dns']], capture_output=True)

        elif brute != 'yes' and reverseDNS == 'yes':
            print('\n[*]', colored('Running the program with reverse lookups and without brute forcing ...', 'red'))
            harvester_output = subprocess.run(['python3', '../theHarvester/theHarvester.py', '-d', target_domain,harvester_options['search_limit'],search_limit,
                                               harvester_options['data_source'], source_choice,
                                               harvester_options['dns_lookup']], capture_output=True)

        elif brute == 'yes' and reverseDNS == 'yes':
            print('\n[*]', colored('Running the program with brute forcing with reverse lookups ...', 'red'))
            harvester_output = subprocess.run(['python3', '../theHarvester/theHarvester.py', '-d', target_domain,harvester_options['search_limit'],search_limit,
                                               harvester_options['data_source'], source_choice,
                                               harvester_options['brute_dns'], harvester_options['dns_lookup']],
                                              capture_output=True)

        else:
            print('\n[*]', colored('Running the program without brute forcing or reverse lookups ...', 'red'))
            harvester_output = subprocess.run(['python3', '../theHarvester/theHarvester.py', '-d', target_domain,harvester_options['search_limit'],search_limit,
                                               harvester_options['data_source'], source_choice],
                                              capture_output=True)

        decoded = harvester_output.stdout.decode('utf-8', errors='ignore').split('[*]')

        if verbose_choice == 'no':
            for i in range(1, len(decoded)):
                # print('Line : ', decoded[i])
                if 'Users found:' in decoded[i][:20]:

                    users_found = decoded[i].split('\n')

                    for n in range(2, len(users_found)):
                        Users.append(users_found[n])

                elif 'Emails found:' in decoded[i][:20]:

                    emails_found = decoded[i].split('\n')

                    for n in range(2, len(emails_found)):
                        Emails.append(emails_found[n])

                elif 'Hosts found:' in decoded[i][:20]:

                    hosts_found = decoded[i].split('\n')

                    for n in range(2, len(hosts_found)):
                        Hosts.append(hosts_found[n])

                elif 'IPs found:' in decoded[i][:20]:

                    ips_found = decoded[i].split('\n')

                    for n in range(2, len(ips_found)):
                        Ips.append(ips_found[n])

                elif 'DNS brute force:' in decoded[i][:30]:

                    brute_hosts_found = decoded[i].split('\n')

                    for n in range(2, len(brute_hosts_found)):
                        Hosts.append(brute_hosts_found[n])

                elif 'Hosts found after reverse lookup:' in decoded[i][:30]:

                    reverse_hosts_found = decoded[i].split('\n')

                    for n in range(2, len(reverse_hosts_found)):
                        Hosts.append(reverse_hosts_found[n])

                urls_found = re.search(r'(http[s]?://[^\s]+)', decoded[i])
                if urls_found is not None:
                    URLs.append(urls_found.group(1))

            # Printing Results
            cprint('\nUsers:', 'yellow', attrs=['bold', 'underline'])
            orderuniq(Users)

            cprint('\nEmails:', 'yellow', attrs=['bold', 'underline'])
            orderuniq(Emails)

            cprint('\nURLs:', 'yellow', attrs=['bold', 'underline'])
            orderuniq(URLs)

            cprint('\nHosts:', 'yellow', attrs=['bold', 'underline'])
            orderuniq(Hosts)

            cprint('\nIPs:', 'yellow', attrs=['bold', 'underline'])
            orderuniq(Ips)

        else:
            for i in decoded:
                print(i)

    except Exception as e:
        print('[*] Sorry an', colored('Error', 'red'), 'has occurred with theHarvester:')
        print(e, '\n')
