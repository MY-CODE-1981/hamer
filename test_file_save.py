import os

dir = "output/mesh/np_savez.npz"
if not os.path.exists(dir): # ディレクトリが存在するか確認
    print("not exist")
else:
    print("already exist")