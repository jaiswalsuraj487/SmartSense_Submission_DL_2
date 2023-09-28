# SmartSense_Submission_DL_2
SmartSense_Submission_DL_2

1. Problem Statement:
● Train a model to predict a person's personality using text.

Steps:
1. I used the data from myPersonality folder file named mypersonality_final.csv as it is revelent to the given problem
2. We saw that we can have 5 types of personality 'Openness': 'cOPN', 'Conscientiousness': 'cCON', 'Extraversion': 'cEXT','Agreeableness': 'cAGR','Neuroticism': 'cNEU'
3. Here we can see person can have multiple personality. So each one of them can be considered as a different output.
4. So we divide the data into 5 parts and train 5 different models for each personality.

5. We will do tfidf vectorization on the posts and then train the model using this vector as imput.


6. I implemnted Random forest and have show accuracy for them
7. Accuries using random forest is better than atleast selecting at random ie 50% accuracy.
8. But we want more accuracy for that I implemnt BERT
9. We can see using only 4 epochs we can achive better results than random forest. 

But due computational time and time constraints i cannot run for more epochs. But by training with more epochs we can attain better accuracy.


I have provided both `submission.py` and `submission.ipynb` files. Please consider both in case either does not execute in your system.



