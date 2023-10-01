# Description of folder GoodReadsDatasets_RedundancyCut:

## Description of folder name:
Take folder GoodReads_309t_245w_3r as an example. This name means that the folder stores the data set crawled from https://goodreads.com. 
The data contains 309 tasks and 245 workers in total, and the task redundancy is close to 3. Other folder names have a similar meaning.
## Folder data source: 
The data set of each folder is divided by the data set GoodReadsDatasets/GoodReads_309t_120415w total data set, 
we randomly select data from this total data set according to different redundancy, and then organize into a new data. 
For example, in the dataset GoodReads_309t_245w_3r, workers and their task labels were randomly selected from the dataset GoodReads_309t_120415w 
according to the standard of task redundancy of 3, and then sorted into a new dataset. 
The tasks, workers, and annotations in this new dataset were extracted from GoodReads_309t_120415w without any other modifications.
## The use of this data set file in the experiment: 
We use this data set in the Effectiveness on Groundtruth Inference with Varying Redundancy experiment part of the paper,
and the results of the experiment come from this dataset.