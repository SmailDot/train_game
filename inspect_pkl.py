import os
import pickle

path = r"d:\NKUST_LAB_Work_Data\traingame\models\vec_normalize_6666.pkl"
if os.path.exists(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    print("Type:", type(data))
    if hasattr(data, "obs_rms"):
        print("Has obs_rms")
        print("Mean shape:", data.obs_rms.mean.shape)
    else:
        print("No obs_rms found")
else:
    print("File not found")
