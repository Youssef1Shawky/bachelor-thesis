import spacy

nlp = spacy.load("en_core_web_md")  # Medium-sized pre-trained model with word vectors 
# (uses word embedded algorithm)

text1 = "one of my favourite hobbies is learning languages . Not only does it boosts your self-confidence but also your communication skills. Imagine being able to speak to all people in the world . What a super power !"
text2 = "Learning languages enhances cognitive abilities such as problem-solving and multitasking, while also fostering cross-cultural understanding and communication skills, facilitating connections in an increasingly globalized world. Additionally, it can broaden career opportunities, offering access to international job markets and facilitating meaningful interactions with people from diverse backgrounds."


doc1 = nlp(text1)
doc2 = nlp(text2)

similarity_score = doc1.similarity(doc2)

print("The similarity between the texts is: ", similarity_score)
