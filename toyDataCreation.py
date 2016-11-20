#GANN toy dataset creation
from __future__ import division
import numpy as np
from random	import randint
import random
import math

#Function for non-picked values
def rngFunc(n, end, start = 0):
    return range(start, n) + range(n+1, end)
	
#Is prime?
def is_prime(n):
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))

#5 digit question & 5 digit answer
numSamples = 3
numPositions = 5

#Declaring empty arrays
quesNpArry = np.zeros((10,numPositions,numSamples))
ansNpArry = np.zeros((10,numPositions,numSamples))

for i in range(numSamples):
	ques = map(int, str(randint(10000,99999)))
	quesArry = np.zeros((10,numPositions))
	ansArry = np.zeros((10,numPositions))
	answr = [ques[0]]
	for j in range(len(ques)):
		#Question Array creation
		quesArry[int(ques[j]),j] = 1
		
		#Answer array creation
		if(j > 0):
			if(is_prime(int(answr[j-1])) == True):
				tmpVar = int(answr[j-1])*2
			else:
				tmpVar = int(answr[j-1])*3
				
			tmpVar = map(int, str(tmpVar))
			answr.append(tmpVar[len(tmpVar)-1])
		
		value = randint(6,9)/10
		#print(answr)
		ansArry[int(answr[j]),j] = value
		remainValue = (1 - value)/3
		possiblePostn = rngFunc(answr[j],10)
		random.shuffle(possiblePostn)
		
		ansArry[possiblePostn[0:3],j] = remainValue
	
	quesNpArry[:,:,i] = quesArry
	ansNpArry[:,:,i] = ansArry
		
print(quesNpArry[:,:,0])
print(ansNpArry[:,:,0])
