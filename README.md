# TransferLearningCodeSearch

This repository contains the implementation details and appendices of the thesis ‘Machine Learning for code search: a systematic review and an industrial feasibility sutdy'. This repository uses three models (found in the 'models' folder) from existing literature. The source code from these three repositories have been adopted and slightly modified where necessary. The following papers and repositories are involved:
- model/CODEnn: https://github.com/guxd/deep-code-search (Gu et al. 2018);
- model/codenn-master: https://github.com/sriniiyer/codenn (Iyer et al. 2016)
- model/CQIL: https://github.com/flyboss/CQIL (Li et al. 2020).

## Datasets
To train and test the models, this implementation uses two publicly available datasets, namely:
- CodeSearchNet: https://github.com/github/CodeSearchNet: (Husian et al. 2015) 
- StaQC: https://github.com/LittleYUYU/StackOverflow-Question-Code-Dataset (Yoa et al. 2018)

Since each model requires different formats for the datasets, the "preprocessing" folder contains the necessary scripts to convert the datasets mentioned above into the appropriate format.
Make sure to add the folders with the original data source to the preprocessing folder. CodeSearchNet contains a folder name ‘python’. For the StaQC dataset, add a folder to the preprocessing folder named ‘StaQC’ and place there the associated pickle files.

'preprocessing_dcs.py' and 'preprocessing_staqc.py' contain the initial preprocessing to put the dataset into the required format for the CQIL and was also used for the preprocessing for CODEnn.


## CODEnn
The folder ‘model/CODEnn’ contains a modified version of the repository https://github.com/guxd/deep-code-search. 
After the datasets are retrieved, they must be added to the folder: ‘data/example’.



## CODE-NN
The trained models on the StaQC can be found in the folder src/model/staqc and the one the CodeSearchNet dataset in scr/model/csn. The model that is wished to be used, should be placed in the scr/model folder.
Note that this repository contains other requirements. In the thesis it is executed in Ubuntu. 
This folder contains a modified version of the repository https://github.com/sriniiyer/codenn. 

After the datasets are retrieved, they must be added to the folder:
-	data/stackoverflow/python
-	and the associated ‘ref.txt’ and ‘dev.txt’ in data/stackoverflow/python/dev

## CQIL
After the datasets are retrieved, they must be added to the folder: ‘data/example/codesearchnet’ and ‘data/example/staqc’. Subsequently, the path to this model should be adjusted in the ‘config.py’ file.

## Baseline models
The folder 'models/baseline_models' contains the baseline models that use the TF-IDF weights and BM25 approach. For the BM25 approach, the Python library 'rank-bm25' (https://github.com/dorianbrown/rank_bm25).

## Reference
- Gu, X., Zhang, H., & Kim, S. (2018, May). Deep code search. In 2018 IEEE/ACM 40th International Conference on Software Engineering (ICSE) (pp. 933-944). IEEE.
- Husain, H., Wu, H. H., Gazit, T., Allamanis, M., & Brockschmidt, M. (2019). Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.
- Iyer, S., Konstas, I., Cheung, A., & Zettlemoyer, L. (2016, August). Summarizing source code using a neural attention model. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 2073-2083).
- Li, W., Qin, H., Yan, S., Shen, B., & Chen, Y. (2020, September). Learning code-query interaction for enhancing code searches. In 2020 IEEE International Conference on Software Maintenance and Evolution (ICSME) (pp. 115-126). IEEE.
- Yao, Z., Weld, D. S., Chen, W. P., & Sun, H. (2018, April). Staqc: A systematically mined question-code dataset from stack overflow. In Proceedings of the 2018 World Wide Web Conference (pp. 1693-1703).
