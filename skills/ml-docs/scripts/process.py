#!/usr/bin/env python3
# /// script
# dependencies = []
# ///
"""
ML Docs URL Resolver — resolves the best documentation URLs to fetch
for any of the 19 supported ML/data science libraries.

Commands:
    list                        List all supported libraries
    resolve --library X --query Y   Resolve fetch URLs for a topic/function
    search-url --library X --query Y  Return just the search URL

Usage:
    uv run scripts/process.py list
    uv run scripts/process.py resolve --library pandas --query "DataFrame.groupby"
    uv run scripts/process.py resolve --library pytorch --query "autograd"
    uv run scripts/process.py search-url --library sklearn --query "cross_val_score"

Supported library aliases (case-insensitive):
    numpy, np
    pandas, pd
    sklearn, scikit-learn, scikit_learn
    matplotlib, mpl, plt
    tensorflow, tf
    keras
    pytorch, torch
    seaborn, sns
    scipy
    statsmodels, sm
    xgboost, xgb
    lightgbm, lgbm
    transformers, huggingface, hf
    opencv, cv2, cv
    nltk
    spacy
    plotly
    dask
    pyspark, spark
    sqlalchemy, sql
    jupyter
"""

import argparse
import json
import sys
from urllib.parse import quote_plus


# ---------------------------------------------------------------------------
# Library catalog
# ---------------------------------------------------------------------------

LIBRARIES = {
    "numpy": {
        "name": "NumPy",
        "aliases": ["np"],
        "base_url": "https://numpy.org/doc/stable/",
        "api_base": "https://numpy.org/doc/stable/reference/generated/numpy.{func}.html",
        "search_url": "https://numpy.org/doc/stable/search.html?q={query}",
        "user_guide": "https://numpy.org/doc/stable/user/index.html",
        "api_index": "https://numpy.org/doc/stable/reference/index.html",
        "topic_urls": {
            "array creation": "https://numpy.org/doc/stable/reference/routines.array-creation.html",
            "indexing": "https://numpy.org/doc/stable/reference/arrays.indexing.html",
            "linear algebra": "https://numpy.org/doc/stable/reference/routines.linalg.html",
            "random": "https://numpy.org/doc/stable/reference/random/index.html",
            "math": "https://numpy.org/doc/stable/reference/routines.math.html",
            "statistics": "https://numpy.org/doc/stable/reference/routines.statistics.html",
            "sorting": "https://numpy.org/doc/stable/reference/routines.sort.html",
            "broadcasting": "https://numpy.org/doc/stable/user/basics.broadcasting.html",
        },
    },
    "pandas": {
        "name": "Pandas",
        "aliases": ["pd"],
        "base_url": "https://pandas.pydata.org/docs/",
        "api_base": "https://pandas.pydata.org/docs/reference/api/pandas.{func}.html",
        "search_url": "https://pandas.pydata.org/docs/search.html?q={query}",
        "user_guide": "https://pandas.pydata.org/docs/user_guide/index.html",
        "api_index": "https://pandas.pydata.org/docs/reference/index.html",
        "topic_urls": {
            "dataframe": "https://pandas.pydata.org/docs/reference/frame.html",
            "series": "https://pandas.pydata.org/docs/reference/series.html",
            "groupby": "https://pandas.pydata.org/docs/reference/groupby.html",
            "indexing": "https://pandas.pydata.org/docs/user_guide/indexing.html",
            "merging": "https://pandas.pydata.org/docs/user_guide/merging.html",
            "reshaping": "https://pandas.pydata.org/docs/user_guide/reshaping.html",
            "io": "https://pandas.pydata.org/docs/reference/io.html",
            "timeseries": "https://pandas.pydata.org/docs/user_guide/timeseries.html",
            "missing data": "https://pandas.pydata.org/docs/user_guide/missing_data.html",
            "visualization": "https://pandas.pydata.org/docs/user_guide/visualization.html",
        },
    },
    "sklearn": {
        "name": "scikit-learn",
        "aliases": ["scikit-learn", "scikit_learn"],
        "base_url": "https://scikit-learn.org/stable/",
        "api_base": "https://scikit-learn.org/stable/modules/generated/{func}.html",
        "search_url": "https://scikit-learn.org/stable/search.html?q={query}",
        "user_guide": "https://scikit-learn.org/stable/user_guide.html",
        "api_index": "https://scikit-learn.org/stable/modules/classes.html",
        "topic_urls": {
            "classification": "https://scikit-learn.org/stable/supervised_learning.html",
            "regression": "https://scikit-learn.org/stable/supervised_learning.html",
            "clustering": "https://scikit-learn.org/stable/modules/clustering.html",
            "preprocessing": "https://scikit-learn.org/stable/modules/preprocessing.html",
            "pipelines": "https://scikit-learn.org/stable/modules/compose.html",
            "cross validation": "https://scikit-learn.org/stable/modules/cross_validation.html",
            "model selection": "https://scikit-learn.org/stable/model_selection.html",
            "feature selection": "https://scikit-learn.org/stable/modules/feature_selection.html",
            "dimensionality reduction": "https://scikit-learn.org/stable/modules/decomposition.html",
            "metrics": "https://scikit-learn.org/stable/modules/model_evaluation.html",
        },
    },
    "matplotlib": {
        "name": "Matplotlib",
        "aliases": ["mpl", "plt"],
        "base_url": "https://matplotlib.org/stable/",
        "api_base": "https://matplotlib.org/stable/api/_as_gen/{func}.html",
        "search_url": "https://matplotlib.org/stable/search.html?q={query}",
        "user_guide": "https://matplotlib.org/stable/users/index.html",
        "api_index": "https://matplotlib.org/stable/api/index.html",
        "topic_urls": {
            "pyplot": "https://matplotlib.org/stable/api/pyplot_summary.html",
            "axes": "https://matplotlib.org/stable/api/axes_api.html",
            "figure": "https://matplotlib.org/stable/api/figure_api.html",
            "subplots": "https://matplotlib.org/stable/gallery/subplots_axes_and_figures/index.html",
            "colors": "https://matplotlib.org/stable/gallery/color/index.html",
            "colormaps": "https://matplotlib.org/stable/gallery/color/colormap_reference.html",
            "animation": "https://matplotlib.org/stable/api/animation_api.html",
            "3d": "https://matplotlib.org/stable/gallery/mplot3d/index.html",
        },
    },
    "tensorflow": {
        "name": "TensorFlow",
        "aliases": ["tf"],
        "base_url": "https://www.tensorflow.org/api_docs/python/tf",
        "api_base": "https://www.tensorflow.org/api_docs/python/tf/{func}",
        "search_url": "https://www.tensorflow.org/s/results?q={query}",
        "user_guide": "https://www.tensorflow.org/guide",
        "api_index": "https://www.tensorflow.org/api_docs/python/tf",
        "topic_urls": {
            "keras": "https://www.tensorflow.org/api_docs/python/tf/keras",
            "data": "https://www.tensorflow.org/api_docs/python/tf/data",
            "layers": "https://www.tensorflow.org/api_docs/python/tf/keras/layers",
            "optimizers": "https://www.tensorflow.org/api_docs/python/tf/keras/optimizers",
            "losses": "https://www.tensorflow.org/api_docs/python/tf/keras/losses",
            "metrics": "https://www.tensorflow.org/api_docs/python/tf/keras/metrics",
            "gradients": "https://www.tensorflow.org/api_docs/python/tf/GradientTape",
            "saved model": "https://www.tensorflow.org/guide/saved_model",
        },
    },
    "keras": {
        "name": "Keras",
        "aliases": [],
        "base_url": "https://keras.io/api/",
        "api_base": "https://keras.io/api/{func}/",
        "search_url": "https://keras.io/search.html?query={query}",
        "user_guide": "https://keras.io/guides/",
        "api_index": "https://keras.io/api/",
        "topic_urls": {
            "layers": "https://keras.io/api/layers/",
            "models": "https://keras.io/api/models/",
            "optimizers": "https://keras.io/api/optimizers/",
            "losses": "https://keras.io/api/losses/",
            "metrics": "https://keras.io/api/metrics/",
            "callbacks": "https://keras.io/api/callbacks/",
            "preprocessing": "https://keras.io/api/data_loading/",
            "saving": "https://keras.io/api/saving/",
            "activations": "https://keras.io/api/layers/activation_layers/",
        },
    },
    "pytorch": {
        "name": "PyTorch",
        "aliases": ["torch"],
        "base_url": "https://docs.pytorch.org/docs/stable/",
        "api_base": "https://docs.pytorch.org/docs/stable/{func}.html",
        "search_url": "https://docs.pytorch.org/docs/stable/search.html?q={query}",
        "user_guide": "https://docs.pytorch.org/docs/stable/index.html",
        "api_index": "https://docs.pytorch.org/docs/stable/index.html",
        "topic_urls": {
            "tensors": "https://docs.pytorch.org/docs/stable/tensors.html",
            "autograd": "https://docs.pytorch.org/docs/stable/autograd.html",
            "nn": "https://docs.pytorch.org/docs/stable/nn.html",
            "optim": "https://docs.pytorch.org/docs/stable/optim.html",
            "dataloader": "https://docs.pytorch.org/docs/stable/data.html",
            "cuda": "https://docs.pytorch.org/docs/stable/cuda.html",
            "distributed": "https://docs.pytorch.org/docs/stable/distributed.html",
            "torchvision": "https://docs.pytorch.org/vision/stable/index.html",
            "saving": "https://docs.pytorch.org/docs/stable/notes/serialization.html",
        },
    },
    "seaborn": {
        "name": "Seaborn",
        "aliases": ["sns"],
        "base_url": "https://seaborn.pydata.org/",
        "api_base": "https://seaborn.pydata.org/generated/seaborn.{func}.html",
        "search_url": "https://seaborn.pydata.org/search.html?q={query}",
        "user_guide": "https://seaborn.pydata.org/tutorial.html",
        "api_index": "https://seaborn.pydata.org/api.html",
        "topic_urls": {
            "distribution": "https://seaborn.pydata.org/tutorial/distributions.html",
            "categorical": "https://seaborn.pydata.org/tutorial/categorical.html",
            "regression": "https://seaborn.pydata.org/tutorial/regression.html",
            "heatmap": "https://seaborn.pydata.org/generated/seaborn.heatmap.html",
            "pairplot": "https://seaborn.pydata.org/generated/seaborn.pairplot.html",
            "facetgrid": "https://seaborn.pydata.org/generated/seaborn.FacetGrid.html",
            "themes": "https://seaborn.pydata.org/tutorial/aesthetics.html",
        },
    },
    "scipy": {
        "name": "SciPy",
        "aliases": [],
        "base_url": "https://docs.scipy.org/doc/scipy/",
        "api_base": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.{func}.html",
        "search_url": "https://docs.scipy.org/doc/scipy/search.html?q={query}",
        "user_guide": "https://docs.scipy.org/doc/scipy/tutorial/index.html",
        "api_index": "https://docs.scipy.org/doc/scipy/reference/index.html",
        "topic_urls": {
            "stats": "https://docs.scipy.org/doc/scipy/reference/stats.html",
            "optimize": "https://docs.scipy.org/doc/scipy/reference/optimize.html",
            "linalg": "https://docs.scipy.org/doc/scipy/reference/linalg.html",
            "signal": "https://docs.scipy.org/doc/scipy/reference/signal.html",
            "interpolate": "https://docs.scipy.org/doc/scipy/reference/interpolate.html",
            "integrate": "https://docs.scipy.org/doc/scipy/reference/integrate.html",
            "sparse": "https://docs.scipy.org/doc/scipy/reference/sparse.html",
            "spatial": "https://docs.scipy.org/doc/scipy/reference/spatial.html",
            "fft": "https://docs.scipy.org/doc/scipy/reference/fft.html",
            "hypothesis testing": "https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests",
        },
    },
    "statsmodels": {
        "name": "statsmodels",
        "aliases": ["sm"],
        "base_url": "https://www.statsmodels.org/stable/",
        "api_base": "https://www.statsmodels.org/stable/generated/statsmodels.{func}.html",
        "search_url": "https://www.statsmodels.org/stable/search.html?q={query}",
        "user_guide": "https://www.statsmodels.org/stable/user-guide.html",
        "api_index": "https://www.statsmodels.org/stable/api.html",
        "topic_urls": {
            "ols": "https://www.statsmodels.org/stable/regression.html",
            "regression": "https://www.statsmodels.org/stable/regression.html",
            "logistic": "https://www.statsmodels.org/stable/discretemod.html",
            "timeseries": "https://www.statsmodels.org/stable/tsa.html",
            "arima": "https://www.statsmodels.org/stable/tsa.html",
            "anova": "https://www.statsmodels.org/stable/anova.html",
            "survival": "https://www.statsmodels.org/stable/duration.html",
            "formula": "https://www.statsmodels.org/stable/example_formulas.html",
        },
    },
    "xgboost": {
        "name": "XGBoost",
        "aliases": ["xgb"],
        "base_url": "https://xgboost.readthedocs.io/en/stable/",
        "api_base": "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
        "search_url": "https://xgboost.readthedocs.io/en/stable/search.html?q={query}",
        "user_guide": "https://xgboost.readthedocs.io/en/stable/tutorials/index.html",
        "api_index": "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
        "topic_urls": {
            "sklearn api": "https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html",
            "parameters": "https://xgboost.readthedocs.io/en/stable/parameter.html",
            "feature importance": "https://xgboost.readthedocs.io/en/stable/tutorials/feature_importance.html",
            "early stopping": "https://xgboost.readthedocs.io/en/stable/tutorials/early_stopping.html",
            "gpu": "https://xgboost.readthedocs.io/en/stable/gpu/index.html",
            "cross validation": "https://xgboost.readthedocs.io/en/stable/python/python_intro.html",
        },
    },
    "lightgbm": {
        "name": "LightGBM",
        "aliases": ["lgbm"],
        "base_url": "https://lightgbm.readthedocs.io/en/stable/",
        "api_base": "https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.{func}.html",
        "search_url": "https://lightgbm.readthedocs.io/en/stable/search.html?q={query}",
        "user_guide": "https://lightgbm.readthedocs.io/en/stable/Python-Intro.html",
        "api_index": "https://lightgbm.readthedocs.io/en/stable/Python-API.html",
        "topic_urls": {
            "parameters": "https://lightgbm.readthedocs.io/en/stable/Parameters.html",
            "sklearn api": "https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html",
            "feature importance": "https://lightgbm.readthedocs.io/en/stable/Python-API.html",
            "dart": "https://lightgbm.readthedocs.io/en/stable/Parameters.html#boosting",
            "categorical": "https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html",
        },
    },
    "transformers": {
        "name": "Hugging Face Transformers",
        "aliases": ["huggingface", "hf"],
        "base_url": "https://huggingface.co/docs/transformers/",
        "api_base": "https://huggingface.co/docs/transformers/model_doc/{func}",
        "search_url": "https://huggingface.co/docs/transformers/search?query={query}",
        "user_guide": "https://huggingface.co/docs/transformers/quicktour",
        "api_index": "https://huggingface.co/docs/transformers/index",
        "topic_urls": {
            "pipeline": "https://huggingface.co/docs/transformers/main_classes/pipelines",
            "tokenizer": "https://huggingface.co/docs/transformers/main_classes/tokenizer",
            "trainer": "https://huggingface.co/docs/transformers/main_classes/trainer",
            "fine-tuning": "https://huggingface.co/docs/transformers/training",
            "bert": "https://huggingface.co/docs/transformers/model_doc/bert",
            "gpt2": "https://huggingface.co/docs/transformers/model_doc/gpt2",
            "t5": "https://huggingface.co/docs/transformers/model_doc/t5",
            "llama": "https://huggingface.co/docs/transformers/model_doc/llama",
            "quantization": "https://huggingface.co/docs/transformers/quantization/overview",
            "peft": "https://huggingface.co/docs/peft/index",
            "datasets": "https://huggingface.co/docs/datasets/index",
        },
    },
    "opencv": {
        "name": "OpenCV",
        "aliases": ["cv2", "cv"],
        "base_url": "https://docs.opencv.org/4.x/",
        "api_base": "https://docs.opencv.org/4.x/search.html?q={func}",
        "search_url": "https://docs.opencv.org/4.x/search.html?q={query}",
        "user_guide": "https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html",
        "api_index": "https://docs.opencv.org/4.x/d1/d0d/modules.html",
        "topic_urls": {
            "image reading": "https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html",
            "color spaces": "https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html",
            "filtering": "https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html",
            "edge detection": "https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html",
            "contours": "https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html",
            "face detection": "https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html",
            "feature detection": "https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html",
            "video": "https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html",
            "transforms": "https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html",
            "thresholding": "https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html",
        },
    },
    "nltk": {
        "name": "NLTK",
        "aliases": [],
        "base_url": "https://www.nltk.org/",
        "api_base": "https://www.nltk.org/api/nltk.{func}.html",
        "search_url": "https://www.nltk.org/search.html?q={query}",
        "user_guide": "https://www.nltk.org/book/",
        "api_index": "https://www.nltk.org/api/nltk.html",
        "topic_urls": {
            "tokenization": "https://www.nltk.org/api/nltk.tokenize.html",
            "stemming": "https://www.nltk.org/api/nltk.stem.html",
            "pos tagging": "https://www.nltk.org/api/nltk.tag.html",
            "parsing": "https://www.nltk.org/api/nltk.parse.html",
            "sentiment": "https://www.nltk.org/api/nltk.sentiment.html",
            "corpora": "https://www.nltk.org/nltk_data/",
            "chunking": "https://www.nltk.org/api/nltk.chunk.html",
            "word frequency": "https://www.nltk.org/api/nltk.probability.html",
        },
    },
    "spacy": {
        "name": "spaCy",
        "aliases": [],
        "base_url": "https://spacy.io/",
        "api_base": "https://spacy.io/api/{func}",
        "search_url": "https://spacy.io/search?q={query}",
        "user_guide": "https://spacy.io/usage",
        "api_index": "https://spacy.io/api",
        "topic_urls": {
            "nlp": "https://spacy.io/api/language",
            "doc": "https://spacy.io/api/doc",
            "token": "https://spacy.io/api/token",
            "span": "https://spacy.io/api/span",
            "pipeline": "https://spacy.io/usage/processing-pipelines",
            "ner": "https://spacy.io/usage/linguistic-features#named-entities",
            "pos": "https://spacy.io/usage/linguistic-features#pos-tagging",
            "training": "https://spacy.io/usage/training",
            "models": "https://spacy.io/usage/models",
            "custom components": "https://spacy.io/usage/processing-pipelines#custom-components",
        },
    },
    "plotly": {
        "name": "Plotly",
        "aliases": [],
        "base_url": "https://plotly.com/python/",
        "api_base": "https://plotly.com/python-api-reference/generated/plotly.{func}.html",
        "search_url": "https://plotly.com/search/?q={query}",
        "user_guide": "https://plotly.com/python/plotly-fundamentals/",
        "api_index": "https://plotly.com/python-api-reference/",
        "topic_urls": {
            "scatter": "https://plotly.com/python/line-and-scatter/",
            "bar": "https://plotly.com/python/bar-charts/",
            "histogram": "https://plotly.com/python/histograms/",
            "heatmap": "https://plotly.com/python/heatmaps/",
            "3d": "https://plotly.com/python/3d-charts/",
            "subplots": "https://plotly.com/python/subplots/",
            "dash": "https://dash.plotly.com/",
            "express": "https://plotly.com/python/plotly-express/",
            "animations": "https://plotly.com/python/animations/",
            "layout": "https://plotly.com/python/figure-factories/",
        },
    },
    "dask": {
        "name": "Dask",
        "aliases": [],
        "base_url": "https://docs.dask.org/en/stable/",
        "api_base": "https://docs.dask.org/en/stable/search.html?q={func}",
        "search_url": "https://docs.dask.org/en/stable/search.html?q={query}",
        "user_guide": "https://docs.dask.org/en/stable/",
        "api_index": "https://docs.dask.org/en/stable/api.html",
        "topic_urls": {
            "array": "https://docs.dask.org/en/stable/array.html",
            "dataframe": "https://docs.dask.org/en/stable/dataframe.html",
            "delayed": "https://docs.dask.org/en/stable/delayed.html",
            "scheduler": "https://docs.dask.org/en/stable/scheduling.html",
            "distributed": "https://distributed.dask.org/en/stable/",
            "bag": "https://docs.dask.org/en/stable/bag.html",
            "best practices": "https://docs.dask.org/en/stable/best-practices.html",
            "dataframe api": "https://docs.dask.org/en/stable/dataframe-api.html",
        },
    },
    "pyspark": {
        "name": "PySpark",
        "aliases": ["spark"],
        "base_url": "https://spark.apache.org/docs/latest/api/python/",
        "api_base": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.{func}.html",
        "search_url": "https://spark.apache.org/docs/latest/api/python/search.html?q={query}",
        "user_guide": "https://spark.apache.org/docs/latest/api/python/getting_started/index.html",
        "api_index": "https://spark.apache.org/docs/latest/api/python/reference/index.html",
        "topic_urls": {
            "sql": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/index.html",
            "dataframe": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html",
            "functions": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html",
            "ml": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml/index.html",
            "streaming": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.streaming/index.html",
            "rdd": "https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html",
            "window": "https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html",
        },
    },
    "sqlalchemy": {
        "name": "SQLAlchemy",
        "aliases": ["sql"],
        "base_url": "https://docs.sqlalchemy.org/en/20/",
        "api_base": "https://docs.sqlalchemy.org/en/20/search.html?q={func}",
        "search_url": "https://docs.sqlalchemy.org/en/20/search.html?q={query}",
        "user_guide": "https://docs.sqlalchemy.org/en/20/tutorial/index.html",
        "api_index": "https://docs.sqlalchemy.org/en/20/genindex.html",
        "topic_urls": {
            "orm": "https://docs.sqlalchemy.org/en/20/orm/index.html",
            "core": "https://docs.sqlalchemy.org/en/20/core/index.html",
            "models": "https://docs.sqlalchemy.org/en/20/orm/mapping_styles.html",
            "queries": "https://docs.sqlalchemy.org/en/20/orm/queryguide/index.html",
            "relationships": "https://docs.sqlalchemy.org/en/20/orm/relationships.html",
            "migrations": "https://alembic.sqlalchemy.org/en/latest/",
            "engine": "https://docs.sqlalchemy.org/en/20/core/engines.html",
            "async": "https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html",
            "pandas integration": "https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html",
        },
    },
    "jupyter": {
        "name": "Jupyter",
        "aliases": [],
        "base_url": "https://docs.jupyter.org/en/latest/",
        "api_base": "https://docs.jupyter.org/en/latest/search.html?q={func}",
        "search_url": "https://docs.jupyter.org/en/latest/search.html?q={query}",
        "user_guide": "https://docs.jupyter.org/en/latest/",
        "api_index": "https://docs.jupyter.org/en/latest/",
        "topic_urls": {
            "magic commands": "https://ipython.readthedocs.io/en/stable/interactive/magics.html",
            "kernels": "https://docs.jupyter.org/en/latest/projects/kernels.html",
            "widgets": "https://ipywidgets.readthedocs.io/en/stable/",
            "nbconvert": "https://nbconvert.readthedocs.io/en/latest/",
            "lab": "https://jupyterlab.readthedocs.io/en/stable/",
            "extensions": "https://jupyterlab.readthedocs.io/en/stable/user/extensions.html",
            "shortcuts": "https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts",
            "display": "https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html",
        },
    },
}


# ---------------------------------------------------------------------------
# Alias resolution
# ---------------------------------------------------------------------------

def _build_alias_map():
    m = {}
    for key, lib in LIBRARIES.items():
        m[key] = key
        for alias in lib["aliases"]:
            m[alias.lower()] = key
    return m


ALIAS_MAP = _build_alias_map()


def resolve_library(name: str) -> dict | None:
    key = ALIAS_MAP.get(name.lower())
    return LIBRARIES.get(key) if key else None


# ---------------------------------------------------------------------------
# URL resolution logic
# ---------------------------------------------------------------------------

def resolve_urls(lib: dict, query: str) -> dict:
    """Given a library dict and a query string, return prioritized fetch targets."""
    q_lower = query.lower().strip()
    results = []

    # 1. Check if query matches a known topic exactly
    for topic, url in lib.get("topic_urls", {}).items():
        if topic in q_lower or q_lower in topic:
            results.append({"priority": 1, "label": f"Topic: {topic}", "url": url})

    # 2. If query looks like a function/class (contains dot or capital), build API url
    if "." in query or query[0].isupper() if query else False:
        func_url = lib["api_base"].replace("{func}", query)
        results.append({"priority": 2, "label": f"API reference: {query}", "url": func_url})

    # 3. Always include search URL
    search_url = lib["search_url"].replace("{query}", quote_plus(query))
    results.append({"priority": 3, "label": "Search results", "url": search_url})

    # 4. Include API index as fallback
    results.append({"priority": 4, "label": "API index", "url": lib["api_index"]})

    # Sort by priority, deduplicate URLs
    seen = set()
    unique = []
    for r in sorted(results, key=lambda x: x["priority"]):
        if r["url"] not in seen:
            seen.add(r["url"])
            unique.append(r)

    return {
        "library": lib["name"],
        "query": query,
        "fetch_in_order": unique,
        "instruction": (
            f"Fetch the URLs below in order. Start with priority 1 (most specific). "
            f"If the first URL returns a 404 or empty content, proceed to the next. "
            f"Synthesize the answer from the first successful result."
        ),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(args):
    rows = []
    for key, lib in LIBRARIES.items():
        aliases = ", ".join(lib["aliases"]) if lib["aliases"] else "-"
        rows.append({"key": key, "name": lib["name"], "aliases": aliases, "base_url": lib["base_url"]})
    print(json.dumps({"libraries": rows, "count": len(rows)}, indent=2))


def cmd_resolve(args):
    lib = resolve_library(args.library)
    if lib is None:
        keys = sorted(LIBRARIES.keys())
        print(json.dumps({"error": f"Unknown library: {args.library}", "supported": keys}), file=sys.stderr)
        sys.exit(1)
    result = resolve_urls(lib, args.query)
    print(json.dumps(result, indent=2))


def cmd_search_url(args):
    lib = resolve_library(args.library)
    if lib is None:
        print(json.dumps({"error": f"Unknown library: {args.library}"}), file=sys.stderr)
        sys.exit(1)
    url = lib["search_url"].replace("{query}", quote_plus(args.query))
    print(json.dumps({"library": lib["name"], "search_url": url}, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ML Docs URL Resolver — find the right docs page to fetch for any ML library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all supported libraries")

    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve fetch URLs for a library + query")
    p_resolve.add_argument("--library", required=True, help="Library name or alias (e.g. pandas, torch, sklearn)")
    p_resolve.add_argument("--query", required=True, help="Function name, class, or topic (e.g. 'DataFrame.groupby')")

    # search-url
    p_search = sub.add_parser("search-url", help="Get just the search URL for a library + query")
    p_search.add_argument("--library", required=True, help="Library name or alias")
    p_search.add_argument("--query", required=True, help="Search query")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "resolve":
        cmd_resolve(args)
    elif args.command == "search-url":
        cmd_search_url(args)


if __name__ == "__main__":
    main()
