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


