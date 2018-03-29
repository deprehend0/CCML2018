import re


def starts_with_number(line):
    words = line.split(' ')
    return words[0].isalnum()


# Read file into memory
quote_lines = []
with open('./quotes/memorable_quotes.txt', mode='r') as file:
    quote_lines = file.readlines()

with open('./quotes/non_memorable_quotes.txt', mode='r') as file:
    lines = file.readlines()
    quote_lines = quote_lines + lines

print(len(quote_lines))
# Retrieve quote lines
last_line_blank = False
quotes = []
for line in quote_lines:
    stripped_line = line.strip().replace('\n', '')
    if not stripped_line:
        last_line_blank = True
    if not last_line_blank:
        if not starts_with_number(stripped_line):
            quotes.append(stripped_line)
    else:
        last_line_blank = False

print(len(quotes))
print(quotes[42])

with open('./quotes/quotes.txt', mode='w+') as file:
    file.write('\n'.join(quotes))
