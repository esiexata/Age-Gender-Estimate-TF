import os
i=0

for _, _, arquivo in os.walk('face_database'):

    for i in range (0,len(arquivo)):
        print(arquivo[i])
