import os


for _, _, arquivo in os.walk('face_database'):

    for i in range (0,len(arquivo)):

        print (len(arquivo))
        print(arquivo[i])




def files_path04(path):
    for p, _, files in os.walk(os.path.abspath(path)):

        for file in files:
            print(os.path.join(p, file))

files_path04('face_database')