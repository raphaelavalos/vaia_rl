import numpy as np
from typing import Optional

class Env:
    def __init__(self, difficulty: Optional[int] = 0):
        #  the true values of each socket
        if difficulty == 1:
            self.true_mean = [3,3.2,4, 3, 5, 6, 7]
            print("# # # # # # # # Medium environment# # # # # # # # ")
        elif difficulty == 2:
            self.true_mean = [4,4.2,4.3,4.5,4.9,5.2,5.4,5.8,6,6.4]
            print("# # # # # # # # Hard environment# # # # # # # # ")        
        else:
            self.true_mean = [3, 5, 7, 9, 11]
            print("# # # # # # # # Easy environment# # # # # # # # ")
        
        print(f"The true means for this run are: \n{self.true_mean}")

    def get_arms(self) -> int:
       return len(self.true_mean)
    
    def reward(self, action: int) -> float:
      return np.random.randn() + self.true_mean[action]
    
    