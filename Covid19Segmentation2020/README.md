# COVID-19 CT Lesion Segmentation (COVID-19-20)

The repository contains scripts used upon participation in COVID-19 Lung CT Lesion Segmentation Challenge 2020 
(COVID-19-20) challenge by [vitali.liauchuk](https://grand-challenge.org/users/vitali.liauchuk/) user.
The challenge was dedicated to the problem of segmentation of lung lesions caused by SARS-CoV-2 infection in 3D CT images.

* Challenge web-page: https://covid-segmentation.grand-challenge.org
* Current framework source codes: [SRC](src)

### Running

The current framework is based on the open-source [MONAI framework](https://github.com/Project-MONAI/tutorials/tree/master/3d_segmentation/challenge_baseline/)
provided by the challenge organizers. 
The basic scripts and setting files are stored in [SRC](src) folder.  
The instructions for running and more detailed descriptions are also there.

Prior to running the scripts, one should
[download](https://covid-segmentation.grand-challenge.org/Download/#) the challenge data.
The data should include `Train`, `Validation` and `Test` sets.
The network training was performed with use of NVIDIA Tesla V100 GPU's with 16 GiB of memory. 

### Results

The results obtained with the proposed approach were submitted multiple times at the challenge
[Validation phase](https://covid-segmentation.grand-challenge.org/evaluation/challenge/leaderboard/).
Moderate results were achieved at this stage with the highest result corresponded to 0.7173 mean Dice score.
The latest submit resulted in 0.7036 mean Dice and 86th position among 200+ submitted results.

During the [Test phase](https://covid-segmentation.grand-challenge.org/evaluation/challenge-second-phase-new-data/leaderboard/)
the proposed approach resulted in 0.6461 mean Dice which corresponded to 8th position among 98 submitted results for non-final ranking 
(see [vitali.liauchuk](https://grand-challenge.org/users/vitali.liauchuk/)).

Information about the final ranking will come soon. 

![Alt text](test_lb_preliminary.png "Test Phase Leaderboard (Preliminary)")

### Acknowledgements

This study was partly supported by the National Institute of Allergy and Infectious Diseases, 
National Institutes of Health, U.S. Department of Health and Human Services, USA through the 
[CRDF](https://www.crdfglobal.org/) project DAA9-19-65987-1 ”Year 8: Belarus TB Database and TB Portal”. 
