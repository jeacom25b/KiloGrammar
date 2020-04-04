import os, time

last_date = -1
while 1:
    date = os.stat("grammar_test.txt").st_mtime *\
        os.stat("kilogrammar.py").st_mtime *\
        os.stat("test_input.txt").st_mtime
    if not date == last_date:
        os.system("clear")
        os.system("python3 kilogrammar.py grammar_test.txt -compile > test.py")
        os.system("python3 test.py test_input.txt -color")
    last_date = date
    time.sleep(0.5)
