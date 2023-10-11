# Project Structure Dntroduction
Through directory explanation, we can quickly understand the structure of the entire project.


| Dictionary | Describution                 |
|------------|------------------------------|
| CATD       | Baseline CATD Algorithm Code |
| GTM        | Baseline GTM Algorithm Code  |
| KDEm       | Baseline KDEm Algorithm Code |
| I_LFCount  | Baseline LFC Algorithm Code  |
| PM_CRH     | Baseline CRH Algorithm Code  |
| CTD        | Baseline CTD Algorithm Code  |
| KAPS       | KAPS Algorithm Code          |
| datasets   | All of the Datasets Files    |

### KAPS Dictionary Detials
The KAPS algorithm is the algorithm we proposed in the paper. In order to adapt to different data set formats, we store two different forms of KAPS algorithm codes in this directory. The core algorithms of the two codes are the same, but they are adapted to different data set formats.
```
KAPS
├─BaselineExperiment ----- This folder contains the code associated with baseline running
│      LaunchBaseline.py ----- This file is a one-click script to launch all baseline code (except CTD), enter the dataset and run this file to get the results of all baseline (except CTD) on the dataset.
│      result_baseline.txt ----- File storing baseline running results
│      
├─DataGraph ----- Figure Code: The code for drawing the experimental results into graphics is stored here: the running result data related to the experiment needs to be input into it to get the graphical result display of the experiment.
│      data_3r_15r_dataset.py ----- Figure: KAPS versus Baselines Comparative Experiment
│      label_distribution.py ----- Figure: Label Distribution of Datasets
│      model_bias.py ----- Figure:Theoretical Deviation and Other Deviations and Figure:Influence of Worker Preferences on Model Accuracy Across Different Datasets.
│      SW_test.py ----- Figure:  Comparison of the Same Metric Changes across Different Datasets.
│      
└─Mine ----- KAPS algorithm code
        EM2.py ----- KAPS algorithm code for all datasets(except twitter dataset)
        EM_Twitter.py ----- KAPS algorithm code for only twitter dataset
        record.txt ----- running record
        result.txt ----- running result
```



-----
# Experimental Code Introduction
All experiments that need to use the KAPS algorithm need to use the experimental data set, calculate the results through the KAPS algorithm, and then put the calculation results into the corresponding experimental code to draw the experimental result graph. The dataset required for each experiment and the plotting code for that experiment are as follows:

## The process of obtaining experimental results：
1. First we need to obtain the corresponding data set of the experiment. The data set of each experiment may be different.
2. Second, we need to put the experimental data set into the KAPS algorithm and other baseline algorithms to calculate the results.
3. Third, we organized the results of the KAPS algorithm and the result data of other baseline algorithms, and obtained the experimental result diagram through the drawing code, which is the diagram shown in the paper.

## Algorithm code location and usage
The KAPS algorithm code is located in the folder "KAPS/Mine/". If the data set is the Douban data set or the GoodReads data set, the "EM2.py" code is used for calculation. If you are using the Twitter data set, use the "EM_Twitter.py" code to calculate.

Then select the corresponding data set according to the specific experiment, and put the result data into the drawing code to get the experimental result graph.

The code of Baseline CATD、GTM、KDEm、LFC、CRH  can be run by a startup script I wrote in the dictionary "EM/BaselineExperiment/LaunchBaseline.py"   

Since the algorithm code for CTD is written in Java, we package each dataset and the CTD algorithm into a jar package, running the corresponding jar package can calculate the corresponding results, these jar packages are located at: "CTD", and the corresponding calculated output is located at: "CTD/log/Tri/CTD/weather/parameter"



## Introduction to datasets and drawing codes for specific experiments

### Effectiveness on Groundtruth Inference with Varying Redundancy.
- **The dataset required for this experiment:**
The dataset for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets_RedundancyCut" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets_RedundancyCut"

- **The location of the drawing code for this experiment:**
The drawing code is located in the directory: "EM/DataGraph/data_3r_15r_dataset.py"

  
### Effectiveness of Kindness.
- **The dataset required for this experiment:**
The Code is located in the directory: "EM/DataGraph/model_bias.py"

- **The location of the drawing code for this experiment:**
The dataset for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_202t_269266w" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_120415w"



### Effectiveness of Kindness.
- **The dataset required for this experiment:**
The Code is located in the directory: "EM/DataGraph/model_bias.py"

- **The location of the drawing code for this experiment:**
The dataset for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets/Douban_202t_269266w" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets/GoodReads_309t_120415w"


### Analysis of Dataset Anomalies.
- **The dataset required for this experiment:**
The Code is located in the directory: "EM/DataGraph/SW_test.py"

- **The location of the drawing code for this experiment:**
The dataset for this experiment is located in "datasets/QuantitativeCrowdsourcing/DouBanDatasets_SWTest" and "datasets/QuantitativeCrowdsourcing/GoodReadsDatasets_SWTest"


### Analysis of User Kindness Distribution.
- **The dataset required for this experiment:**
Figure(b): This result leads us to the result of running the "Gephi" software with the dataset: "datasets/QuantitativeCrowdsourcing/TwitterDatasets/Twitter_394t_46486w/G_Twitter1.csv" and "datasets/QuantitativeCrowdsourcing/TwitterDatasets/Twitter_394t_46486w/H_Twitter1.csv"

- **The location of the drawing code for this experiment:**
Figure(a): The Code is located in the directory: "EM/DataGraph/label_distribution.py"

-----

# Dataset Introduction
We have collected three datasets from [Douban](https://www.douban.com/)、[GoodReads](https://www.goodreads.com/), and [Twitter](https://twitter.com/) using web crawlers.

**Douban**: We collected and organized the long review information (1-5 points) of the Top 250 movies and the social relationship attributes between the commenting users. We also used the expert ratings of movies from [metacritic](https://www.metacritic.com/) as the ground truth for this dataset.

**GoodReads**: We collected and organized some user reviews (1-5 points) from the 2022 Best Books list and the social relationship attributes between the commenting users. The ground truth for this dataset is the average rating of the authors.

**Twitter**: In this dataset, we crawled and organized comments under President Biden’s tweets on political topics and the social relationships between commenting users. We used a Bert model to score the sentiment of the comments, with an output value in the range of [0,1]. The larger the value, the more positive it is, and vice versa.

We have classified the datasets into two folders: complete_dataset and split_dataset.

The attributes of the complete dataset are shown in the table below:
| Dataset name  | items  | workers  | labels  | social connections  |
| ------------ | ------------ | ------------ | ------------ | ------------ |
| Douban  |  202 |  269266 | 437478  | 2340905  |
|  GoodReads | 309  | 120415  | 224837  | 293298  |
|  Twitter | 394  | 46486  | 63265  | 1065302  |

The complete datasets for Douban, GoodReads, and Twitter are stored in the complete_dataset folder. Only the split datasets for Douban and GoodReads are stored in the split_dataset folder. The splitting rule is to gradually increase worker redundancy from 3 to 15 for each task, with priority given to workers with high social influence. The calculation of social influence is as follows: social influence = (degree of node +1) / total number of nodes.

The description of each file and column in each dataset i

The description of each file and column in each dataset is as follows:

* **E.csv**: Social influence file (worker id, social influence)
* **G.csv**: Social relationship file (follower, fan)
* **T.csv**: Task category file (each column represents a category, if it is equal to one then it belongs to that category)
* **truth.csv**: Ground truth file (task id, ground truth)
* **Y.csv**: Worker task file (worker id, task id, label)
* **Y2.csv**: Worker task file (worker id, task id, label, ground truth)
* **result_a.csv**: This file is the result of the code running, and we will put the result of the code running here to facilitate the relevant experiment. This file has nothing to do with the lab setup and is for my own use only.
* **result_h.csv**: This file is the result of the code running, and we will put the result of the code running here to facilitate the relevant experiment. This file has nothing to do with the lab setup and is for my own use only.
* **result_phi.csv**: This file is the result of the code running, and we will put the result of the code running here to facilitate the relevant experiment. This file has nothing to do with the lab setup and is for my own use only.
* **result_R.csv**: This file is the result of the code running, and we will put the result of the code running here to facilitate the relevant experiment. This file has nothing to do with the lab setup and is for my own use only.
* **t2lpd.csv**: This file is applied in part of the drawing code for the predicted values for each task. If you need to reproduce the implementation, you need to regenerate the file from the corresponding dataset.

-----
# More Detilas in Dataset Introduction
All of our dataset files in the folder datasets/QuantitativeCrowdsourcing, this folder has 7 subfolders, represent the 7 different datasets, and each has a markdown readme file folder, Used to explain the details of this folder dataset. For ease of reading, I have organized the introduction documentation of these datasets as follows:
## Dataset DouBanDatasets/Douban_202t_269266w  Describption:
This folder contains datasets crawled from Douban. 
The folder Douban_202t_269266w is the raw data, and this naming means that the data structure in the dataset in this folder contains 202 tasks and 269266 workers.
We collected and organized the long review information (1-5 points) of the Top 250 movies and the social relationship attributes between the commenting users. We also used the expert ratings of movies from metacritic as the ground truth for this dataset.

## Dataset DouBanDatasets_RedundancyCut Describption:
### Description of folder name:
Take folder Douban_202t_110w_3r as an example. This name means that the folder stores the dataset crawled from Douban.com. The data contains 202 tasks and 110 workers in total, and the task redundancy is close to 3. Other folder names have a similar meaning.
### Folder data source: 
The dataset of each folder is divided by the dataset DouBanDatasets/Douban_202t_269266w total dataset, we randomly select data from this total dataset according to different redundancy, and then organize into a new data. For example, in the dataset Douban_202t_110w_3r, workers and their task labels were randomly selected from the dataset DouBanDatasets/Douban_202t_269266w according to the standard of task redundancy of 3, and then sorted into a new dataset. The tasks, workers, and annotations in this new dataset were extracted from DouBanDatasets/Douban_202t_269266w without any other modifications.
### The use of this dataset file in the experiment: 
We use this dataset in the Effectiveness on Groundtruth Inference with Varying Redundancy experiment part of the paper, and the results of the experiment come from this dataset.

## Dataset DouBanDatasets_SWTest Describption:
### Description of folder name
We apply the Shapiro–Wilk Test method to analyze the normality in the frequency statistical test of the datasets.       Since the original dataset consists of a large proportion of workers who have only annotated a few tasks, it does not meetthe detection conditions of the Shapiro-Wilk test algorithm.        Hence,workers with fewer than 5 total label tasks are removed from the dataset DouBanDatasets/Douban_202t_269266w.       Therefore, we obtain a new dataset, where the number of label tasks for each worker is greater than 5, satisfying the detection requirements.        This dataset is Douban_7271w_0%.
Next, we process the resulting dataset Douban_7271w_0% and get ten datasets.
the Shapiro-Wilk test is conducted for all label results of each worker,and the corresponding 𝑤 value for each worker is calculated.             The 𝑤 values are sorted in ascending order, and datasets are generated by continuously excluding workers with small 𝑤 values, at a 5% scale, leading to ten datasets.              For example, Douban_6906w_5% dataset means sort in ascending order of w values, excluding those achieved by the top 5% of the smallest workers.          Similarly, Douban_6543w_10% represents the values sorted in ascending order of w, excluding the top 10% of workers.
Through this rule, we extract and sort out ten datasets from the original dataset Douban_7271w_0%, which have not been modified except the data size.
### Folder data source
The dataset Douban_7271w_0% is extracted and collated from the dataset DouBanDatasets/Douban_202t_269266w. The ten datasets of Douban_6906w_5%~Douban_6906w_50% are obtained from the Douban_7271w_0% dataset by continuously excluding the people with the lowest w value in proportion.
### The use of this dataset file in the experiment
This Dataset was used in the experimental Analysis of Dataset Anomalies, figure: "Comparison of the Same Metric Changes across
Different Datasets" represent the results of this experiment.

## Dataset GoodReadsDatasets/GoodReads_309t_120415w Describption:
This folder contains datasets crawled from GoodReads. The folder GoodReads_309t_120415w is the raw data, 
and this naming means that the data structure in the dataset in this folder contains 309 tasks and 120415 workers.
We collected and organized some user reviews (1-5 points) from the 2022 Best Books list and the social relationship attributes between the commenting users. The ground truth for this dataset is the average rating of the authors.

## Dataset GoodReadsDatasets_RedundancyCut Describption:
### Description of folder name:
Take folder GoodReads_309t_245w_3r as an example. This name means that the folder stores the dataset crawled from https://goodreads.com. 
The data contains 309 tasks and 245 workers in total, and the task redundancy is close to 3. Other folder names have a similar meaning.
### Folder data source: 
The dataset of each folder is divided by the dataset GoodReadsDatasets/GoodReads_309t_120415w total dataset, 
we randomly select data from this total dataset according to different redundancy, and then organize into a new data. 
For example, in the dataset GoodReads_309t_245w_3r, workers and their task labels were randomly selected from the dataset GoodReads_309t_120415w 
according to the standard of task redundancy of 3, and then sorted into a new dataset. 
The tasks, workers, and annotations in this new dataset were extracted from GoodReads_309t_120415w without any other modifications.
### The use of this dataset file in the experiment: 
We use this dataset in the Effectiveness on Groundtruth Inference with Varying Redundancy experiment part of the paper,
and the results of the experiment come from this dataset.

## Dataset GoodReadsDatasets_SWTest Describption:
### Description of folder name
We apply the Shapiro–Wilk Test method to analyze the normality in the frequency statistical test of the datasets.
Since the original dataset consists of a large proportion of workers who have only annotated a few tasks, 
it does not meet the detection conditions of the Shapiro-Wilk test algorithm.
Hence,workers with fewer than 5 total label tasks are removed from the dataset GoodReads_309t_120415w. 
Therefore, we obtain a new dataset, where the number of label tasks for each worker is greater than 5, satisfying the detection requirements. This dataset is GoodReads_5702_309t_0%.
Next, we process the resulting dataset GoodReads_5702_309t_0% and get ten datasets.
the Shapiro-Wilk test is conducted for all label results of each worker,and the corresponding 𝑤 value for each worker is calculated.
The 𝑤 values are sorted in ascending order, and datasets are generated by continuously excluding workers with small 𝑤 values, 
at a 5% scale, leading to ten datasets. 
For example, GoodReads_5415_309t_5% dataset means sort in ascending order of w values, excluding those achieved by the top 5% of the smallest workers.          Similarly, Douban_6543w_10% represents the values sorted in ascending order of w, excluding the top 10% of workers.
Through this rule, we extract and sort out ten datasets from the original dataset GoodReads_5702_309t_0%, which have not been modified except the data size.
### Folder data source
The dataset GoodReads_5702_309t_0% is extracted and collated from the dataset GoodReads_309t_120415w.
The ten datasets of GoodReads_5415_309t_5% ~ GoodReads_2850_308t_50% are obtained from the GoodReads_5702_309t_0% dataset by continuously excluding the people with the lowest w value in proportion.
### The use of this dataset file in the experiment
This Dataset was used in the experimental Analysis of Dataset Anomalies, figure: "Comparison of the Same Metric Changes across
Different Datasets" represent the results of this experiment.

## Dataset TwitterDatasets/Twitter_394t_46486w Describption:
### Description of folder name
This dataset is the twitter dataset, which we crawled from the Twitter website using crawler technology. In this dataset,  we crawled and organized comments under President Biden’s tweets on political topics and the social relationships  between commenting users. We used a Bert model to score the sentiment of the comments,  with an output value in the range of [0,1]. The larger the value, the more positive it is, and vice versa.
And this dataset include 394 tasks and 46486 workers.
### The use of this dataset file in the experiment
- We obtained the experimental result by using the software Gephi and running the dataset Twitter_394t_46486w: "Influence of Worker Preferences on Model Accuracy.
Across Different Datasets (b) Kindness Map in Twitter ", By importing G_Twitter1.csv and H_Twitter1.csv files or only import  Twitter_Great2.gephi in the folder Twitter_394t_46486w into the software Gephi, the effect in the experiment diagram can be achieved.






