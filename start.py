import os
os.system("pip install -r requirements.txt")
os.chdir("./server/")
os.system("python server.py")
input("Hit Enter to exit")