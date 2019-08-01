# Gutenburg NLP Analysis:

### Blog Link:
https://medium.com/rapids-ai/show-me-the-word-count-3146e1173801


### Objective: Show case nlp capabilties of nvstrings+cudf

### Pre-Processing :
* filter punctuation
* to_lower
* remove stop words (from nltk corpus)
* remove multiple spaces with one
* remove leading and trailing spaces    
    
### Word Count: 
* Get Frequency count for the whole dataset
* Compare word count for two authors (Albert Einstein vs Charles Dickens )
* Get Word counts for all the authors

### Encode the word-count for all authors into a count-vector

We do this in two steps:

1. Encode the string Series using `top 20k` most used `words` in the Dataset which we calculated earlier.
    * We encode anything not in the series to string_id = `20_000` (`threshold`)


2. With the encoded count series for all authors, we  create an aligned word-count vector for them, where:
    * Where each column corresponds to a `word_id` from the the `top 20k words`
    * Each row corresponds to the `count vector` for that author
    
    
### Find the nearest authors using the count-vector:
* Fit a knn
* Find the authors nearest to each other in the count vector space
* Decrease dimunitonality using UMAP
* Find the authors nearest to each other in the latent space
