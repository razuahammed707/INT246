
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits

digits = load_digits()
import pylab as pl 
pl.gray() 
pl.matshow(digits.images[0]) 
pl.show()


digits.images[0] #Each element represents the pixel of our greyscale image. The value ranges from 0 to 255 for an 8 bit pixel.





images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)

import random
from sklearn import ensemble

#Define variables
n_samples = len(digits.images)
x = digits.images.reshape((n_samples, -1))
y = digits.target

#Create random indices 
sample_index=random.sample(range(len(x)),len(x)/5) #20-80
valid_index=[i for i in range(len(x)) if i not in sample_index]

#Sample and validation images
sample_images=[x[i] for i in sample_index]
valid_images=[x[i] for i in valid_index]

#Sample and validation targets
sample_target=[y[i] for i in sample_index]
valid_target=[y[i] for i in valid_index]

#Using the Random Forest Classifier
classifier = ensemble.RandomForestClassifier()

#Fit model with sample data
classifier.fit(sample_images, sample_target)

#Attempt to predict validation data
score=classifier.score(valid_images, valid_target)
print 'Random Tree Classifier:\n' 
print 'Score\t'+str(score)

i=78

pl.gray() 
pl.matshow(digits.images[i]) 
pl.show() 
outi=classifier.predict(x[[i]])
print("OUTPUT IS",outi)
