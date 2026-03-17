Sentiment Analysis on IMDB Reviews - Maasticht University NLP Lab task

Built a sentiment analysis tool that classifies movie reviews as positive or negative using the Stanford IMDB dataset from Hugging Face.

Approached the problem using:

1. Hand-crafted features — extracted review length, punctuation counts, and sentiment word counts using an opinion lexicon. Iteratively improved the feature set by adding polarity ratios, exclamation/question mark counts, and uppercase word frequency.

2. Bag-of-Words + TF-IDF — built a BOW representation, tested different vocabulary sizes (1000, 1500, 2000), applied TF-IDF weighting, and combined it with the hand-crafted features.

3. BERT embeddings — used a pre-trained bert-base-cased model from Hugging Face. Extracted sentence embeddings via the CLS token and by averaging hidden states of non-special tokens, then fed them into a logistic regression classifier.

Compared multiple classifiers (logistic regression, linear SVM, SGD) and evaluated using F1, balanced accuracy, confusion matrix, and ROC-AUC.

Results on test set: BERT performed best (~88% balanced accuracy), BOW close behind (~87%), hand-crafted features lowest as expected.
