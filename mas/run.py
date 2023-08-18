import pandas as pd
from nltk.translate.gleu_score import sentence_gleu
import experiments

annotation_df = pd.read_csv("./data/boundingbox_data.csv", error_bad_lines=False)

distance_fn = lambda x,y: 1 - (sentence_gleu([x.split(" ")], y.split(" ")) + sentence_gleu([y.split(" ")], x.split(" "))) / 2

translation_experiment = experiments.RealExperiment(eval_fn=None,
                                                    label_colname="annotation",
                                                    item_colname="item", uid_colname="uid",
                                                    distance_fn=distance_fn)

translation_experiment.setup(annotation_df)
translation_experiment.train()