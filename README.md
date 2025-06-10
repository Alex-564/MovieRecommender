# MOVIE RECOMMENDER

## BACKGROUND AND TECHNICAL JARGON

This project is a KNN based AI movie recommender that uses the IMDB dataset to explore similar movies based on preset features. This was a passion project, I love movies and also I really enjoy experimenting with AI so the blend was kinda perfect to practice and create a project with.

The project doesnt use the whole IMDB dataset due to size and memory constraints, instead opting for a more filtered version. The filters used can be found in "imdb_filterset.ipynb" (more on that below).

### HOW IT WORKS

In this project we construct a KNN model of movies and explore similar neighbours which act as recommendations due to similarities between them.

The model is constructed off of many features which are extracted from the main dataset, standardized/vectorized, and converted into a sparse matrix for memory concerns, enabling it to scale well.

Working off of KNN the model builds its space data directly, plotting all of the moves according to their weighted features in space. When a movie is selected by the user, the model uses COSINE similarity to scan for the k nearest neighbours to that particular data point.


## FILES AND CONTENTS

1. "imdb_filterset.ipynb":

This was included to show the filterings done to the full IMDB dataset as it could not entirely be used due to size limitations.
The user is more than welcome to reset their own filterings and apply it to the original IMDB dataset - Just make sure that you download the correct one !!
--> The one I used in my development can be found here : 


2. "MovieRecommender_builder.ipynb":

This is the Jupyter notebook used to extract, weight, standardize, and vectorize all the features used in the recommender model, it includes the step by step for extracting the features, and then in the final step it includes a code segment to export the model to a pickle file: this can be found in "models/...".
This is open to extensibility to add more features to the model (Just ensure that the changes are included in the pickle file).

Note on weightings:
- Weightings are really important to the output of the model, my weightings were derived from heuristic values outputted from Chat GPT as a rough guide, they are by no means perfect.

The suggested weightings for type are as follows:
- Genres: Medium to high weight (1.0-1.5), Genres are broad categories—good at filtering (e.g., horror vs. romcom), but bad at nuance. Movies from different genres rarely feel similar.

- Keywords: High weight (1.5-2.5), Keywords are often richer, more specific tags. They can reflect themes, setting, plot devices—exactly what people care about in movie similarity.

- Vote Average: Low weight (0.2-0.5), Averages say more about quality/popularity, not content.


3. "Movie Recommender.py":

This is the main script for the project, it takes an input, loads the pickle model, and tests for reccomended outputs.
This is the ONLY file that should be ran unless youre looking to create a new model entirely.


## HOW TO RUN

Run python3 MovieRecommender.py 