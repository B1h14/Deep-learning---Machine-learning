Basic implementation of the ROUGE-1 metric. 
The file contains a solution for the problem : https://www.deep-ml.com/problems/152
the function rouge_1_score in the fie rouge_score.py contains an implementation of the metric. 
Given the two texts , the candidate and the reference text , the function returns a dict containing the precision , recall and f1-score for the metric.

Rouge-1 Score description : 
This metric is used to evaluate the similarity of a generated summary by comparing it to a reference summary. It measures the overlap of unigrams (individual words) between the generated summary and the reference. 
A higher ROUGE-1 score indicates that more words in the generated summary match those in the reference, suggesting better content similarity.