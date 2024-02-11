import joblib
import vigenere

clf = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

answers=[]

# Classification of new strings
newTexts = []
i=0
while (i<100):
    newTexts.append(vigenere.randomText())
    answers.append("random")
    i+=1
i=0
while (i<100):
    newTexts.append(vigenere.ciphertext())
    answers.append("cipher")
    i+=1

# Vectorize the new texts using the same TF-IDF vectorizer
newTextsTfidf = vectorizer.transform(newTexts)

# Predict
new_predictions = clf.predict(newTextsTfidf)

correct=0
incorrect=0
cipherPredictedAsRandom=0
randomPredictedAsCipher=0
cipherPredictedAsCipher=0
randomPredictedAsRandom=0

for text, prediction, answer in zip(newTexts, new_predictions, answers):
    if prediction==answer:
        correct+=1
        if prediction=="cipher":
            cipherPredictedAsCipher+=1
        else:
            randomPredictedAsRandom+=1
    else:
        incorrect+=1
        if prediction=="cipher":
            randomPredictedAsCipher+=1
        else:
            cipherPredictedAsRandom+=1
    
print("Correct: ", str(correct) + " ("+str(correct/(incorrect+correct)*100)[0:5]+"%),   Incorrect: ", str(incorrect)+ " ("+str(incorrect/(incorrect+correct)*100)[0:5]+"%)")
print("Cipher predicted as cipher: ", cipherPredictedAsCipher)
print("Random predicted as random: ", randomPredictedAsRandom)
print("Cipher predicted as random: ", cipherPredictedAsRandom)
print("Random predicted as Cipher: ", randomPredictedAsCipher)
