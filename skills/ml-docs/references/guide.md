# ML Docs Reference Guide

## Library Aliases

| Key | Full Name | Aliases |
|-----|-----------|---------|
| `numpy` | NumPy | `np` |
| `pandas` | Pandas | `pd` |
| `sklearn` | scikit-learn | `scikit-learn`, `scikit_learn` |
| `matplotlib` | Matplotlib | `mpl`, `plt` |
| `tensorflow` | TensorFlow | `tf` |
| `keras` | Keras | _(none)_ |
| `pytorch` | PyTorch | `torch` |
| `seaborn` | Seaborn | `sns` |
| `scipy` | SciPy | _(none)_ |
| `statsmodels` | statsmodels | `sm` |
| `xgboost` | XGBoost | `xgb` |
| `lightgbm` | LightGBM | `lgbm` |
| `transformers` | Hugging Face Transformers | `huggingface`, `hf` |
| `opencv` | OpenCV | `cv2`, `cv` |
| `nltk` | NLTK | _(none)_ |
| `spacy` | spaCy | _(none)_ |
| `plotly` | Plotly | _(none)_ |
| `dask` | Dask | _(none)_ |
| `pyspark` | PySpark | `spark` |
| `sqlalchemy` | SQLAlchemy | `sql` |
| `jupyter` | Jupyter | _(none)_ |

---

## Direct Topic URLs (skip the script for these common lookups)

### NumPy
| Topic | URL |
|-------|-----|
| Array creation | https://numpy.org/doc/stable/reference/routines.array-creation.html |
| Indexing & slicing | https://numpy.org/doc/stable/reference/arrays.indexing.html |
| Linear algebra | https://numpy.org/doc/stable/reference/routines.linalg.html |
| Random number generation | https://numpy.org/doc/stable/reference/random/index.html |
| Math functions | https://numpy.org/doc/stable/reference/routines.math.html |
| Statistics | https://numpy.org/doc/stable/reference/routines.statistics.html |
| Broadcasting | https://numpy.org/doc/stable/user/basics.broadcasting.html |
| Full API reference | https://numpy.org/doc/stable/reference/index.html |

### Pandas
| Topic | URL |
|-------|-----|
| DataFrame API | https://pandas.pydata.org/docs/reference/frame.html |
| Series API | https://pandas.pydata.org/docs/reference/series.html |
| GroupBy | https://pandas.pydata.org/docs/reference/groupby.html |
| Merging / joining | https://pandas.pydata.org/docs/user_guide/merging.html |
| Reshaping / pivot | https://pandas.pydata.org/docs/user_guide/reshaping.html |
| I/O (read_csv, read_sql, etc.) | https://pandas.pydata.org/docs/reference/io.html |
| Time series | https://pandas.pydata.org/docs/user_guide/timeseries.html |
| Missing data | https://pandas.pydata.org/docs/user_guide/missing_data.html |
| 2.0 migration guide | https://pandas.pydata.org/docs/whatsnew/v2.0.0.html |

### scikit-learn
| Topic | URL |
|-------|-----|
| All estimators (API index) | https://scikit-learn.org/stable/modules/classes.html |
| Preprocessing | https://scikit-learn.org/stable/modules/preprocessing.html |
| Pipelines | https://scikit-learn.org/stable/modules/compose.html |
| Cross-validation | https://scikit-learn.org/stable/modules/cross_validation.html |
| Model evaluation / metrics | https://scikit-learn.org/stable/modules/model_evaluation.html |
| Feature selection | https://scikit-learn.org/stable/modules/feature_selection.html |
| Dimensionality reduction | https://scikit-learn.org/stable/modules/decomposition.html |
| Clustering | https://scikit-learn.org/stable/modules/clustering.html |
| Supervised learning overview | https://scikit-learn.org/stable/supervised_learning.html |

### Matplotlib
| Topic | URL |
|-------|-----|
| pyplot functions | https://matplotlib.org/stable/api/pyplot_summary.html |
| Axes API | https://matplotlib.org/stable/api/axes_api.html |
| Figure API | https://matplotlib.org/stable/api/figure_api.html |
| Subplots | https://matplotlib.org/stable/gallery/subplots_axes_and_figures/index.html |
| Colors | https://matplotlib.org/stable/gallery/color/index.html |
| Colormaps | https://matplotlib.org/stable/gallery/color/colormap_reference.html |
| Animation | https://matplotlib.org/stable/api/animation_api.html |
| 3D plots | https://matplotlib.org/stable/gallery/mplot3d/index.html |

### TensorFlow
| Topic | URL |
|-------|-----|
| tf.keras API | https://www.tensorflow.org/api_docs/python/tf/keras |
| Layers | https://www.tensorflow.org/api_docs/python/tf/keras/layers |
| Optimizers | https://www.tensorflow.org/api_docs/python/tf/keras/optimizers |
| Losses | https://www.tensorflow.org/api_docs/python/tf/keras/losses |
| tf.data pipeline | https://www.tensorflow.org/api_docs/python/tf/data |
| GradientTape | https://www.tensorflow.org/api_docs/python/tf/GradientTape |
| Saving / loading models | https://www.tensorflow.org/guide/saved_model |

### Keras (standalone 3.x)
| Topic | URL |
|-------|-----|
| Layers | https://keras.io/api/layers/ |
| Models | https://keras.io/api/models/ |
| Optimizers | https://keras.io/api/optimizers/ |
| Losses | https://keras.io/api/losses/ |
| Metrics | https://keras.io/api/metrics/ |
| Callbacks | https://keras.io/api/callbacks/ |
| Saving / serialization | https://keras.io/api/saving/ |
| Guides | https://keras.io/guides/ |

### PyTorch
| Topic | URL |
|-------|-----|
| Tensor operations | https://docs.pytorch.org/docs/stable/tensors.html |
| Autograd | https://docs.pytorch.org/docs/stable/autograd.html |
| nn module | https://docs.pytorch.org/docs/stable/nn.html |
| Optimizers | https://docs.pytorch.org/docs/stable/optim.html |
| DataLoader / Dataset | https://docs.pytorch.org/docs/stable/data.html |
| CUDA | https://docs.pytorch.org/docs/stable/cuda.html |
| Distributed training | https://docs.pytorch.org/docs/stable/distributed.html |
| Saving models | https://docs.pytorch.org/docs/stable/notes/serialization.html |
| torchvision | https://docs.pytorch.org/vision/stable/index.html |

### Seaborn
| Topic | URL |
|-------|-----|
| API reference | https://seaborn.pydata.org/api.html |
| Distribution plots | https://seaborn.pydata.org/tutorial/distributions.html |
| Categorical plots | https://seaborn.pydata.org/tutorial/categorical.html |
| Regression plots | https://seaborn.pydata.org/tutorial/regression.html |
| Themes / aesthetics | https://seaborn.pydata.org/tutorial/aesthetics.html |

### SciPy
| Topic | URL |
|-------|-----|
| Statistical tests | https://docs.scipy.org/doc/scipy/reference/stats.html |
| Optimization | https://docs.scipy.org/doc/scipy/reference/optimize.html |
| Linear algebra | https://docs.scipy.org/doc/scipy/reference/linalg.html |
| Signal processing | https://docs.scipy.org/doc/scipy/reference/signal.html |
| Interpolation | https://docs.scipy.org/doc/scipy/reference/interpolate.html |
| Integration | https://docs.scipy.org/doc/scipy/reference/integrate.html |
| Sparse matrices | https://docs.scipy.org/doc/scipy/reference/sparse.html |
| Spatial algorithms | https://docs.scipy.org/doc/scipy/reference/spatial.html |
| FFT | https://docs.scipy.org/doc/scipy/reference/fft.html |

### statsmodels
| Topic | URL |
|-------|-----|
| OLS regression | https://www.statsmodels.org/stable/regression.html |
| Logistic / discrete models | https://www.statsmodels.org/stable/discretemod.html |
| Time series (ARIMA, SARIMA) | https://www.statsmodels.org/stable/tsa.html |
| ANOVA | https://www.statsmodels.org/stable/anova.html |
| Formula interface | https://www.statsmodels.org/stable/example_formulas.html |
| Full API | https://www.statsmodels.org/stable/api.html |

### XGBoost
| Topic | URL |
|-------|-----|
| Python API | https://xgboost.readthedocs.io/en/stable/python/python_api.html |
| sklearn estimator | https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html |
| All parameters | https://xgboost.readthedocs.io/en/stable/parameter.html |
| Feature importance | https://xgboost.readthedocs.io/en/stable/tutorials/feature_importance.html |
| Early stopping | https://xgboost.readthedocs.io/en/stable/tutorials/early_stopping.html |
| GPU support | https://xgboost.readthedocs.io/en/stable/gpu/index.html |

### LightGBM
| Topic | URL |
|-------|-----|
| Python API | https://lightgbm.readthedocs.io/en/stable/Python-API.html |
| All parameters | https://lightgbm.readthedocs.io/en/stable/Parameters.html |
| sklearn API | https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html |
| Advanced topics | https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html |

### Hugging Face Transformers
| Topic | URL |
|-------|-----|
| Pipeline API | https://huggingface.co/docs/transformers/main_classes/pipelines |
| Tokenizer API | https://huggingface.co/docs/transformers/main_classes/tokenizer |
| Trainer API | https://huggingface.co/docs/transformers/main_classes/trainer |
| Fine-tuning guide | https://huggingface.co/docs/transformers/training |
| BERT | https://huggingface.co/docs/transformers/model_doc/bert |
| GPT-2 | https://huggingface.co/docs/transformers/model_doc/gpt2 |
| T5 | https://huggingface.co/docs/transformers/model_doc/t5 |
| LLaMA | https://huggingface.co/docs/transformers/model_doc/llama |
| Quantization | https://huggingface.co/docs/transformers/quantization/overview |
| PEFT / LoRA | https://huggingface.co/docs/peft/index |
| Datasets library | https://huggingface.co/docs/datasets/index |

### OpenCV
| Topic | URL |
|-------|-----|
| Python tutorials root | https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html |
| Reading/writing images | https://docs.opencv.org/4.x/db/deb/tutorial_display_image.html |
| Color space conversions | https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html |
| Filtering / blurring | https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html |
| Edge detection (Canny) | https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html |
| Contours | https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html |
| Face detection | https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html |
| Feature detection | https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html |
| Video capture | https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html |
| Geometric transforms | https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html |
| Thresholding | https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html |

### NLTK
| Topic | URL |
|-------|-----|
| Tokenization | https://www.nltk.org/api/nltk.tokenize.html |
| Stemming / lemmatization | https://www.nltk.org/api/nltk.stem.html |
| POS tagging | https://www.nltk.org/api/nltk.tag.html |
| Parsing | https://www.nltk.org/api/nltk.parse.html |
| Sentiment (VADER) | https://www.nltk.org/api/nltk.sentiment.html |
| Corpora / data | https://www.nltk.org/nltk_data/ |
| NLTK Book (full tutorial) | https://www.nltk.org/book/ |

### spaCy
| Topic | URL |
|-------|-----|
| Language (nlp) | https://spacy.io/api/language |
| Doc object | https://spacy.io/api/doc |
| Token object | https://spacy.io/api/token |
| Span object | https://spacy.io/api/span |
| Processing pipelines | https://spacy.io/usage/processing-pipelines |
| Named entity recognition | https://spacy.io/usage/linguistic-features#named-entities |
| POS tagging | https://spacy.io/usage/linguistic-features#pos-tagging |
| Training | https://spacy.io/usage/training |
| Models | https://spacy.io/usage/models |
| Custom pipeline components | https://spacy.io/usage/processing-pipelines#custom-components |

### Plotly
| Topic | URL |
|-------|-----|
| Plotly Express (high-level) | https://plotly.com/python/plotly-express/ |
| Scatter / line charts | https://plotly.com/python/line-and-scatter/ |
| Bar charts | https://plotly.com/python/bar-charts/ |
| Histograms | https://plotly.com/python/histograms/ |
| Heatmaps | https://plotly.com/python/heatmaps/ |
| 3D charts | https://plotly.com/python/3d-charts/ |
| Subplots | https://plotly.com/python/subplots/ |
| Animations | https://plotly.com/python/animations/ |
| Dash (web apps) | https://dash.plotly.com/ |
| Python API reference | https://plotly.com/python-api-reference/ |

### Dask
| Topic | URL |
|-------|-----|
| Dask Array | https://docs.dask.org/en/stable/array.html |
| Dask DataFrame | https://docs.dask.org/en/stable/dataframe.html |
| Dask DataFrame API | https://docs.dask.org/en/stable/dataframe-api.html |
| Delayed (task graph) | https://docs.dask.org/en/stable/delayed.html |
| Schedulers | https://docs.dask.org/en/stable/scheduling.html |
| Distributed cluster | https://distributed.dask.org/en/stable/ |
| Dask Bag | https://docs.dask.org/en/stable/bag.html |
| Best practices | https://docs.dask.org/en/stable/best-practices.html |

### PySpark
| Topic | URL |
|-------|-----|
| Getting started | https://spark.apache.org/docs/latest/api/python/getting_started/index.html |
| SQL / DataFrame API | https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html |
| Built-in functions | https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html |
| Window functions | https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/window.html |
| MLlib | https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml/index.html |
| Structured Streaming | https://spark.apache.org/docs/latest/api/python/reference/pyspark.streaming/index.html |
| RDD API | https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html |

### SQLAlchemy
| Topic | URL |
|-------|-----|
| Tutorial (ORM) | https://docs.sqlalchemy.org/en/20/tutorial/index.html |
| ORM overview | https://docs.sqlalchemy.org/en/20/orm/index.html |
| Mapping styles | https://docs.sqlalchemy.org/en/20/orm/mapping_styles.html |
| Query guide | https://docs.sqlalchemy.org/en/20/orm/queryguide/index.html |
| Relationships | https://docs.sqlalchemy.org/en/20/orm/relationships.html |
| Engine / connections | https://docs.sqlalchemy.org/en/20/core/engines.html |
| Async SQLAlchemy | https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html |
| Alembic migrations | https://alembic.sqlalchemy.org/en/latest/ |

### Jupyter
| Topic | URL |
|-------|-----|
| Magic commands (IPython) | https://ipython.readthedocs.io/en/stable/interactive/magics.html |
| Kernels | https://docs.jupyter.org/en/latest/projects/kernels.html |
| ipywidgets | https://ipywidgets.readthedocs.io/en/stable/ |
| nbconvert | https://nbconvert.readthedocs.io/en/latest/ |
| JupyterLab | https://jupyterlab.readthedocs.io/en/stable/ |
| JupyterLab extensions | https://jupyterlab.readthedocs.io/en/stable/user/extensions.html |
| Keyboard shortcuts | https://jupyterlab.readthedocs.io/en/stable/user/interface.html#keyboard-shortcuts |
| IPython display | https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html |

---

## Library Selection Guide

**I need to... — use this library**

| Task | Best library |
|------|-------------|
| Numerical arrays, matrix math | NumPy |
| Tabular data manipulation, ETL | Pandas |
| Classical ML (regression, classification, clustering) | scikit-learn |
| Gradient boosting (structured/tabular data) | XGBoost or LightGBM |
| Deep learning, research flexibility | PyTorch |
| Deep learning, production / TFX ecosystem | TensorFlow + Keras |
| NLP, LLMs, fine-tuning transformers | Hugging Face Transformers |
| Quick NLP (tokenize, stem, basic corpus work) | NLTK |
| Production NLP pipelines, NER, dependency parsing | spaCy |
| Computer vision preprocessing | OpenCV |
| Static statistical plots | Matplotlib or Seaborn |
| Interactive / web plots | Plotly |
| Statistical modeling, p-values, ARIMA | statsmodels |
| Scientific computing, hypothesis tests, signal | SciPy |
| Data larger than RAM (parallelized Pandas/NumPy) | Dask |
| Big data, distributed SQL | PySpark |
| Database ORM, SQL queries from Python | SQLAlchemy |
| Interactive notebooks | Jupyter |

---

## Version Notes

| Library | Current stable | Key breaking version |
|---------|---------------|----------------------|
| NumPy | 2.x | 2.0 removed many deprecated aliases |
| Pandas | 2.x | 2.0 removed `append`, copy-on-write in 2.0+ |
| scikit-learn | 1.5.x | Pipeline `set_output` API added in 1.2 |
| PyTorch | 2.x | `torch.compile` added in 2.0 |
| TensorFlow | 2.16+ | Keras 3 decoupled from tf.keras |
| Keras | 3.x | Backend-agnostic (TF, JAX, PyTorch) |
| Transformers | 4.x | `pipeline` task names stable since 4.0 |
| Pandas | 2.x | `DataFrame.append` removed — use `pd.concat` |
