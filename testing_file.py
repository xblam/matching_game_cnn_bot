import argparse

def main():
    parser = argparse.ArgumentParser(description='This script demonstrates how to use short and long option names.')
    parser.add_argument("-db", "--hostname", help="Database name")
    parser.add_argument("-u", "--username", help="User name")
    parser.add_argument("-p", "--password", help="Password")
    parser.add_argument("-size", "--size", help="Size", type=int)   
    # Parse the arguments
    args = parser.parse_args()

    # Use the parsed arguments
    print('Hostname:', args.hostname)
    print('Username:', args.username)
    print('Password:', args.password)
    print('Size:', args.size)

if __name__ == '__main__':
    main()