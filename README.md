# Dictionary Explaination

| Dictionary | Describution        |
|------------|---------------------|
| CATD       | Baseline CATD Code  |
| GTM        | Baseline GTM Code   |
| KDEm       | Baseline KDEm Code  |
| I_LFCount  | Baseline LFC Code   |
| PM_CRH     | Baseline CRH Code   |
| CTD        | Baseline CTD Code   |
| EM         | KARS Code           |
| datasets   | All of the Datasets |


# Experiment Code
All experiments that need to use the Kars algorithm need to use the experimental data set, through the Kars algorithm to calculate the results, and then put the calculated results into the corresponding experimental code to draw the experimental results map. The data sets needed for each experiment and the drawing code for that experiment are as follows:

1. Effectiveness on Groundtruth Inference with Varying Redundancy.
- The KARS Code is located in the directory: "EM/DataGraph/data_3r_15r_dataset.py"
- The KARS dataset for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets_RedundancyCut" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets_RedundancyCut"
- The code of Baseline CATD、GTM、KDEm、LFC、CRH  can be run by a startup script I wrote in the dictionary "EM/BaselineExperiment/LaunchBaseline.py"   
- Since the algorithm code for CTD is written in Java, we package each data set and the CTD algorithm into a jar package, running the corresponding jar package can calculate the corresponding results, these jar packages are located at: "CTD", and the corresponding calculated output is located at: "CTD/log/Tri/CTD/weather/parameter"


2. Effectiveness of Kindness.
- The Code is located in the directory: "EM/DataGraph/model_bias.py"
- The data set for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_202t_269266w" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_120415w"


3. Effectiveness of Preferences.
- The Code is located in the directory: "EM/DataGraph/model_bias.py"
- The data set for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_202t_269266w" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_120415w"


4. Analysis of Dataset Anomalies.
- The Code is located in the directory: "EM/DataGraph/SW_test.py"
- The data set for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets_SWTest" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets_SWTest"


5. Analysis of User Kindness Distribution.
- Figure(a): The Code is located in the directory: "EM/DataGraph/label_distribution.py"
- Figure(b): This result leads us to the result of running the "Gephi" software with the dataset: "datasets/QuantitativeCrowdsourcing/TwitterDatasets/Twitter_394t_46486w/G_Twitter1.csv" and "datasets/QuantitativeCrowdsourcing/TwitterDatasets/Twitter_394t_46486w/H_Twitter1.csv"

# Dataset Introduction
We have collected three datasets from [Douban](https://www.douban.com/)、[GoodReads](https://www.goodreads.com/), and [Twitter](https://twitter.com/) using web crawlers.

Douban: We collected and organized the long review information (1-5 points) of the Top 250 movies and the social relationship attributes between the commenting users. We also used the expert ratings of movies from [metacritic](https://www.metacritic.com/) as the ground truth for this dataset.

GoodReads: We collected and organized some user reviews (1-5 points) from the 2022 Best Books list and the social relationship attributes between the commenting users. The ground truth for this dataset is the average rating of the authors.

Twitter: In this dataset, we crawled and organized comments under President Biden’s tweets on political topics and the social relationships between commenting users. We used a Bert model to score the sentiment of the comments, with an output value in the range of [0,1]. The larger the value, the more positive it is, and vice versa.

We have classified the datasets into two folders: complete_dataset and split_dataset.

The attributes of the complete dataset are shown in the table below:
| Dataset name  | items  | workers  | labels  | social connections  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| Douban  |  202 |  269266 | 437478  | 2340905  |
|  GoodReads | 309  | 120415  | 224837  | 293298  |
|  Twitter | 394  | 46486  | 63265  | 1065302  |

The complete datasets for Douban, GoodReads, and Twitter are stored in the complete_dataset folder. Only the split datasets for Douban and GoodReads are stored in the split_dataset folder. The splitting rule is to gradually increase worker redundancy from 3 to 15 for each task, with priority given to workers with high social influence. The calculation of social influence is as follows: social influence = (degree of node +1) / total number of nodes.

The description of each file and column in each dataset i
