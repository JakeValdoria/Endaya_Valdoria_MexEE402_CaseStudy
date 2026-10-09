# MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Endaya, Earl Jasper | 22-07503 | MEXE-4103 |
| Valdoria, Jake | 22-04213 | MEXE-4103 |

## Notebook links

| Chapter | ENDAYA, EARL JASPER | VALDORIA, JAKE C. |
|---|---|---|
| Ch1_2_3 | [link]() | https://colab.research.google.com/drive/1GkqRrTyZr_UV3hyHVI-6e0kbsA3HCNuZ?usp=drive_link |
| Ch4 | [link]() | https://colab.research.google.com/drive/1Mi5WvPe8LXEJp8n2EcsagkcIEkgplwqU?usp=drive_link |
| Ch5 | [link]() | https://colab.research.google.com/drive/1sb6_CvSzFqQ4ryc4Zl5qFxGMqV9OOu9-?usp=drive_link |
| Ch6 | https://colab.research.google.com/drive/1hNMW1RrOAd1KXjnpJh1NkxGPGkQfU-OU?usp=sharing | [link]() |
| Ch7 | https://colab.research.google.com/drive/1odb9Wxh5QqOhW1FgYzdhGoBWhXFTmNTd?usp=sharing | [link]() |
| Ch8 | https://colab.research.google.com/drive/1baglD9jl-tluXKfV64TcBtSlEOPi7SAU?usp=sharing | [link]() |
| Ch9 | https://colab.research.google.com/drive/1VDzzgYo7vIbvAfj_M77BgXd7EC7-JGHy?usp=sharing | [link]() |

## What we learned

In chapter 1, we learned that the data should be organized and have no error before using it in machine learning. We realized that even with much data this will not be valuable if it is messy, if there is an error or if there is any missing data. Also we understand that preprocessing data is one of the important things before using machine learning. 

In chapter 2, we understand and learned that it is important to know first what is the condition and what is in the dataset. We realized we need to check the rows , columns, missing values, or data types. We also realized that a simple checking of the dataset can let you know more about the data. 

In this chapter 3 gives us a better understanding about being safe in handling missing data or data that is not needed. We also realized that not all the missing values have the same way of fixing it. Also we learned that there are information that can be removed because it is not useful in analysis.

The most important part of chapter 4 that we have learned is that we can make new information from existing data. We also realized that by combining variables, we can easily see the patterns in the data. Moreover, from simple existing data we can create new useful data for machine learning.

This chapter 5 gives us a better idea why scaling numerical data is important. Having a big difference in values of the features can affect the result of the learning machine model. We also realized that scaling is not always necessary because it depends on the data we are working with. 

The chapter 6 helped us understand that an outlier is more than just a big number, because it can change how the whole dataset looks. In the sample, the value 100 pulled the mean up to 26.5, while the median stayed at 17.5, so we learned that the median is steadier when outliers are present. We were surprised that the Z-score method did not flag 100 (its Z-score was 2.615, just under the cutoff of 3), while the IQR method did. From this, we learned that IQR may be a good first choice when a dataset is small.

The chapter 7 taught us that feature selection does not always give one single answer. The three methods gave three different results: the filter kept 3 features, RFECV kept assignments completed, and Lasso kept a different group of 3, including extracurricular activities. We learned that each method asks a different question: is the feature related to the target, does the model score better with it, and does the model still need it while training. It also reminded us to read outputs carefully, because the filter's list included final grade, which is the value we are trying to predict.

The chapter 8 showed us that a pipeline is like writing a recipe once and reusing it every time. What helped us most was understanding the order of the steps: missing values are filled first, and then the data is scaled. Age had 177 missing values (about 20% of the 891 passengers), so the imputer step was important. We also liked that ColumnTransformer lets us choose exactly which columns to process which are Age and Fare

Lastly, This chapter 9 brought together everything from the earlier chapters using the Titanic data. We also really appreciated the graphs in this chapter, because they helped us understand the data much better than numbers alone. The survival count plot showed that only about 38% of the 891 passengers survived. The gender and passenger class plots made it easy to compare who was more likely to survive. The fare boxplot and the correlation heatmap gave us a quick picture of how the columns relate to each other. We are thankful that these visuals were included, since they made the results easier to understand and gave us a good way to double-check our work.

## Errors we found

While working through the notebooks, we noticed a few small things that we would like to respectfully share. We may have misunderstood some parts, and we are happy to be corrected.

Chapters 1 to 5, we found no major errors that would affect the overall results of the activities.

Chapter 6: In the first outlier example, the notebook says that the number 100 is clearly an outlier. When we ran the cell, though, it showed an empty list, which means the code did not find any outlier. We think this is because the score for 100 was 2.615, which is a little under the limit of 3 that the notebook uses. So in this example, the first method misses it, but the second method (IQR) catches it. We think it would be clearer if the note said this, or if the limit were lowered to 2.

Chapter 7: In the first method for choosing features, the final list of “useful features” also includes final grade. We think this should not be there, because final grade is the answer we want to predict, not one of the features we use to predict it. It appears because every column matches itself perfectly, so it passes the filter automatically. If we remove it first, the list should only have study hours, assignments completed, and class participation.

Chapter 8: We did not find anything that needed correcting in this notebook. The steps ran correctly, and the results matched the explanations. We may have missed something, so we are open to any feedback.

Chapter 9: We noticed two small things here. First, the two graphs that compare age before and after grouping do not seem to show what their titles say. The “before” graph is drawn after the ages were already changed into Child, Adult, and Elderly, so it shows only three bars. The “after” graph uses the third column of the cleaned data, which is the Embarked column, not Age. Second, the introduction says that Fare would be adjusted and PassengerId removed, but we could not find any cell that does this. We think these could be fixed by keeping the original ages in a separate column and adding the two missing steps.

## Note on AI tools

We used AI to translate English into Tagalog so that the lessons would be easier to understand and help us develop a better understanding of each topic. AI also provided different examples that helped us understand more clearly what we were doing in each chapter.

We used also a few different AI tools for this case study. The one we found most helpful was Claude, because in our experience it handles coding-related questions very well. It helped us to notice parts of the notebooks that were easy to miss by eye, such as outputs that did not match their explanations. It also helped us to understand how each term and function in the code works, so we could see why the program gives the output it does. Then checked these points against the notebook outputs ourselves, and we can explain them in our own words.

## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.
