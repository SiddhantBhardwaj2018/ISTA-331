files = [f + '.txt' for f in 
         ['aaou', 'gotg1', 'gotg2a', 'gotg2b', 'gw', 'saguaro']]
for fname in files:
    for i, line in enumerate(open(fname, 'rb').readlines(), 1):
        for j, ch in enumerate(line, 1):
            if (ch < 32 or ch > 126) and ch not in [10, 13]:
                print(fname, 'line:', i, 'col:', j, ch, chr(ch))
            