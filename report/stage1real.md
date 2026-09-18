# Stage 1, Pacome fromager, Machine Learning project

Problem formulation:

The aim of this project is to find the expected rating gain from contestants in competitive programming contests. The website used to find data, Codeforces, work with an elo rating similar to chess, and each contest affect this rating. The goal is to determine the elo gain from this contest given the past performances of an individual. It's most certainly impossible to reach a perfect score in that kind of task, but I will try to make it work as much as possible.
The dataset consists of 11,917,755 datapoints that I collected, all representing a performance of some individual in a given contests. There are 974,594 accounts in total, and 1796 contests, ranging from 2010-02-19 to 2026-09-13.

The data is made of a lot of flags, showing for instance in which division is the contestant competing, and a lot of continuous data, like the current elo of the individual, etc. This will be detailed more later, but there are 18 features, among which 9 continuous ones, 8 binary ones, and an integer.

Methods : 

I first of all scrapped every contest I could find on codeforces.com, through their API. There was 2,145 of them, but 349 had no rated participants, thus I just removed them and had 1,796 contests left. The API gave 7 out of the 18 features that are included, and the rest of them were constructed to make more sense of the data. This was especially important because I was planning on using Linear Regression as the first method, and I knew that adding new features, which give some non-linearity, was going to be useful to get better results.

What also matters a lot in determining the next performance in an upcoming contests is how many contests the individual took part in, because codeforces gives a serious boost to the first few contests. Accounts created since 2020 have a mean of +270, +190, +100, +65, +25 of elo gain over the first 6 contests.

Feature selection:

It used to work the other way before 2020 though: accounts started at 1500 elo directly and actually tended to lose some on their first contest instead (around -60 on average). This convinced me that whether an account is at its very first contest matters a lot on its own, on top of just how many contests it played before, so I made a binary flag for that (is_debut) and kept it alongside the contest count. For the 8% of rows that are an actual debut, all the "past contest" features obviously don't exist yet, so I just set them to 0, which combined with the flag doesn't bias anything since multiplying by 0 cancels out whatever the model would have done with that feature.

To check that the features I picked actually make sense and not just guesses, I looked at how each one correlates with the elo change on the training rows. The strongest ones are the elo relative to the field average (-0.58), the debut flag (+0.47) and the log of the past contest count (-0.46). Some barely correlate at all, like the number of days since the last contest (-0.04) or the average past rank (-0.03), but I kept them anyway since correlation only looks at one feature at a time and won't catch interactions between features.

| feature | corr. | feature | corr. |
|---|---|---|---|
| rating_vs_field | -0.58 | prior_best_rank | +0.12 |
| is_debut | +0.47 | division_Div4 | +0.13 |
| log1p_prior_contest_count | -0.46 | division_Div3 | +0.10 |
| prior_avg_rating_change | +0.37 | division_Div2 | -0.10 |
| prior_rating_change_std | -0.34 | division_Div1 | -0.05 |
| field_avg_rating | -0.27 | days_since_last_contest | -0.04 |
| rating_trend_last3 | +0.29 | prior_avg_rank | -0.03 |
| log_num_participants | +0.22 | division_Global | -0.01 |
| | | division_Div1+2 | -0.01 |
| | | division_ICPC | -0.01 |

Something I made sure NOT to use as a feature is the rank obtained in the contest itself, or the new rating. Codeforces computes the elo change almost directly from the rank, so giving that to the model would basically be handing it the answer instead of making it predict anything.

Model:

I went with Linear Regression to start with. When I plotted the elo change against the elo relative to the field, the relationship looked roughly linear (slope around -0.18), the weights stay easy to read afterwards, and fitting it is basically instant even on close to 10 million rows since it has a closed-form solution. I know it can't really capture the non-linear boost new accounts get though, so that's a limitation I'm aware of for this first method.

Loss function:

I trained it by minimizing the squared error (normal equations), which is the standard loss for a regression problem like this one. To actually judge how good it is I look at both the mean absolute error, directly in elo points, and the root mean squared error.

Validation:

For splitting the data I didn't do it randomly. Codeforces changed its rating system over time (see the 2020 change above) and the whole population of players evolved a lot too, so a random split would basically let the model peek at the future while training. Instead I sorted everything chronologically and cut it into 80% training, 10% validation and 10% test, making sure each cut falls between two contests and never in the middle of one.

| set | rows | share | period |
|---|---|---|---|
| training | 9,536,412 | 80.0% | 2010-02-19 to 2025-01-12 |
| validation | 1,189,681 | 10.0% | 2025-01-17 to 2025-10-10 |
| test | 1,191,662 | 10.0% | 2025-10-12 to 2026-09-13 |

Right now the linear regression gets a validation MAE of 54.27, against 101.88 for just predicting the average elo gain every time, so it's already clearly doing something, even if there's obviously a hard limit to how well this can ever work, since a lot of what decides a performance just can't be known ahead of time.

| | MAE | RMSE |
|---|---|---|
| training | 56.38 | 73.38 |
| validation | 54.27 | 72.76 |
| baseline (predict training mean) | 101.88 | 147.06 |

Use of AI:

I used Claude Code for this project, mostly to write the scraping and feature-building scripts, to fit the linear regression, and to help me spot things in the data I wouldn't have caught on my own, like the whole 2020 elo bonus change and a bug in an earlier version of my division-parsing code. I decided what to build and what to keep, checked the numbers myself, and wrote this report on my own.

Appendix:

The code for this project (scraping, feature building, and training the linear regression) is available at https://github.com/pacomef/machine-learning-aalto.

