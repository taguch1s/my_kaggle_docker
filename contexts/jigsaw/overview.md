# Overview
If you’ve ever had a comment taken down on Reddit and wondered “why?”, you’re not alone. Each subreddit has its own set of guidelines, and trying to understand individual subreddit moderation can feel like chaos.

In this competition, you’ll bring some ‘comment sense’ to the table and work with real data to build models that predict which rule (if any) a comment may have broken.

## Description
Your task is to create a binary classifier that predicts whether a Reddit comment broke a specific rule. The dataset comes from a large collection of moderated comments, with a range of subreddit norms, tones, and community expectations.

The rules you’ll be working with are based on actual subreddit guidelines, but the dataset itself is drawn from older, unlabeled content. A small labeled dev set has been created to help you get started.

This is a chance to explore how machine learning can support real-world content moderation, particularly in communities with unique rules and norms.

## Background
Inspired by the work of our colleagues Deepak Kumar, Yousef AbuHashem, and Zakir Durumeric where large language models were deployed to try to guess the reasons that moderators used to remove comments. This work builds upon the work of Eshwar Chandrasekharan and Eric Gilbert which collected a set of millions of moderated comments.

This several-year-old dataset is unlabeled. It is accompanied by a list of hypothetical rules—derived from real rules on a variety of subreddits—to help identify potential comment violations.

## Rules Classification
Participants have access to a small subset of the data, which can be used as a dev resource. This information is suitable for use as training data or for few-shot examples. The remainder of the labels will be used, in a 30%:70% to form the public and private test sets.

## Evaluation
### scoring
Submissions are evaluated on column-averaged AUC.

### Submission File
For each row_id in the test set, you must predict the probability that a comment violates a given rule. The file should contain a header and have the following format:

```csv
row_id,rule_violation
2029,0.5
2030,0.67
2031,0.1
etc.
```

---

# Dataset Description
The dataset provides instances of comments that may or may not have violated a specific rule on a subreddit.

The training dataset contains only two rules. The test dataset contains additional rules that models must be able to generalize to. (The number of unseen rules is not specified as part of the competition.)

## Files
- train.csv - the training dataset
  - body - the text of the comment
  - rule - the rule the comment is judged to be in violation of
  - subreddit - the forum the comment was made in
  - positive_example_{1,2} - examples of comments that violate the rule
  - negative_example_{1,2} - examples of comments that do not violate the rule
  - rule_violation - the binary target
- test.csv - the test dataset; your objective is to predict the probability of a rule_violation. NOTE: The test dataset contains additional rules that are not seen in the in the training data, so models must be flexible to unseen rules.
- sample_submission.csv - a sample submission file in the correct format.
