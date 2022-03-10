# Transformers Based Lie Detectors: Deceptive Opinions Detection Using Transformers on Single and Multiple Domains
## Lie_Detector_NLP_transformers

The project was made for the **097215 Natural Language Processing** course at the Technion - Israel Institute of Technology.

_lie_detector.py_ - running the experiment mention in our paper: **"Transformers Based Lie Detectors: Deceptive Opinions Detection Using Transformers on Single and Multiple Domains"**. the results of this experiment are output into txt files which we parse into an Excel (csv) file using _parser_results.py_ .
_create_mix_files.py_ - scripts for creating our datasets.

in order to run the experiment by your self:
1. run create_mix_files.py once (choose which datasets to begin with and which output datasets you will get)
2. run lie_detector.py - choose which datasets and hyperparameters to run it with (we recommend to split the run for different datasets since it may take a long time to run all of them in one run)
3. if you wish to collect the experiment outputs in a convenient way - run parser_results.py which will collect all the data into a single csv file (which can be easily read using Office-Excel)

notice we added an Excel file called _results_all_parsed4084.xlsx_ that sum-up the cases we experimented in our paper