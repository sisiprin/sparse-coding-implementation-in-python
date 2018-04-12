import scipy.io as sio

def loadmat(filename, key):
    f = sio.loadmat(filename)
    #keyname = filename[:-5]
    return f[key]