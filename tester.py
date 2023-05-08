import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', type = float, default = 8, help='Number for printing')

args = parser.parse_args()
temp_a = args.b

print(f'The values are {temp_a}', '\n')
