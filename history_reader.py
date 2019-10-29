import pickle as p
from pprint import pprint as pp

f = open("history.p","rb")
hist = p.load(f)
pp(hist)
f.close()