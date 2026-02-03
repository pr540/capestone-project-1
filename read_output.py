import os
with open('compare_output.txt', 'rb') as f:
    content = f.read().decode('utf-16le', errors='ignore')
print(content)
