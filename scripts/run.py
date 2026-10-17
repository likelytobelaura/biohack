import sys,recon
for n in sys.argv[1:]: print('=====',n); getattr(recon,n)()
