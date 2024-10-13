
from Functions_and_Declarations import *



#-----------------------------------------------------------SET COLLECTION VARIABLES-----------------------------------------------------------------------------------------------------------------
sign = 'p'
initial_rep = 0
repititions = 10
num_frames = 30
folder_name = 'MP_TRAINING_DATA'
prep_time = 25  # in deciseconds, so 10 would amount to 1 second.

#-----------------------------------------------------------RUN CODE----------------------------------------------------------------------------------------------------------------------------------               
# PRESS 'q' TO QUIT
# PRESS 'p' TO PAUSE FOR 5 SECONDS
data_collection(sign, initial_rep, repititions, num_frames, folder_name, prep_time)
cap.release()
cv2.destroyAllWindows()
    
    #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    #plt.show()
    
    
