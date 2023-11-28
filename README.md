# Content-based book recommendation system

The goal of this project was to develop a book recommendation system. Three content-based recommendation systems were created and benchmarked against each other. These systems were implemented using TF-IDF, word2vec, and Sentence-BERT, matching users to books based on their reading history.

## Scope

The scope of the study was restricted to the [Good Book](https://github.com/zygmuntz/goodbooks-10k) dataset. This dataset contains six million ratings for the ten thousand most popular books (those with the most ratings as of 2017). There are also individual book ratings by users, books marked to read by the users, book metadata (author, year, etc.) and tags/shelves/genres.

## File structure

<pre>
|- data/
   |- raw/
|- notebooks/
   |- content_based_recommendation.ipynb
   |- figures/
|- book-content-based-recommender/
   |- custom_funcs.py
|- .gitignore
|- LICENSE
|- README.md
</pre>