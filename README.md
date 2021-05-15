# finalproj
Final project of Spl. topics in NLP

Following changes in the code base (15/05/2021 17:08)
1. Added the ipynb for tunning the retrieval engine
2. The logits were converted to probality and second value is taken as the 
   classification probablity in ranker.py
   
3. PQE.py has two parameters MAX_DF and MIN_DF to select the terms which are 
   not frequent, barring a few occurace words. Removing too frequent words 
   as well
   
4. Interrogative pronouns are now added to ignore list
5. Ranker has been extended to run on all the files
