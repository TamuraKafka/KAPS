# Description of folder DouBanDatasets_RedundancyCut:

## Description of folder name:
Take folder Douban_202t_110w_3r as an example. This name means that the folder stores the data set crawled from Douban.com. The data contains 202 tasks and 110 workers in total, and the task redundancy is close to 3. Other folder names have a similar meaning.
## Folder data source: 
The data set of each folder is divided by the data set DouBanDatasets/Douban_202t_269266w total data set, we randomly select data from this total data set according to different redundancy, and then organize into a new data. For example, in the dataset Douban_202t_110w_3r, workers and their task labels were randomly selected from the dataset DouBanDatasets/Douban_202t_269266w according to the standard of task redundancy of 3, and then sorted into a new dataset. The tasks, workers, and annotations in this new dataset were extracted from DouBanDatasets/Douban_202t_269266w without any other modifications.
## The use of this data set file in the experiment: 
We use this data set in the Effectiveness on Groundtruth Inference with Varying Redundancy experiment part of the paper, and the results of the experiment come from this data set.


