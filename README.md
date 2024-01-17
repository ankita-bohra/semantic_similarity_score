# Semantic Similarity Score

**How does Semantic Similarity Score questions is gving the matching score between two sentences?**  
It works on Semantic Sentence Similarity based algorithm. 

Use Cases: We often get sentences which have similar meaning in a survey dataset or may be in the answer sheets of students. We may easily remove redundant information by comparing the score of similarity in survey comments. And it can also be helpful for checking the correctness of answers by comparing the similarity score of a student’s answer.

Proposed Method:
•	Extra embedding removal, Tokenizing sentences, calculation of mean_pool = Sum of embeddings/ Count of Mask
•	Build Model (Model used -> sentence-transformers/bert-base-nli-mean-tokens)
•	Train Model	
•	Test Model
•	Deployment of model using FastAPI

Tools & Technology:
•	NLP (Natural Language Processing)
•	Machine Learning
•	Python








