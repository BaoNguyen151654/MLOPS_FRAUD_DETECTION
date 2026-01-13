# MLops Pipelines For Real-time Credit Fraud Detection
---
## 1/ Overview:
Just imagine you are a normal customer at PNC Bank.
Your credit card life is very predictable: you just use it to order food online, usually under $70, and you’re loyal to your hometown (you basically never leave it). If your credit card had a personality, it would be "boring but reliable".

Then one night, at 2:45 AM, while you’re peacefully asleep, your credit card decides it’s time for an adventure.

💥 $6,350 spent at an online electronics store.

💥 Another big purchase a few minutes later.

💥 Both occurring nearly 1000 miles apart.

The fraud detection system at PNC then raises an eyebrow.

“This doesn’t look like our late-night pizza order…”

Within seconds, the system flags the transactions, blocks the card, and sends you a message asking if you’ve somehow turned into a midnight tech enthusiast. You reply NO, your card is saved, and your credit card goes back to its quiet, food-ordering life.

**=> It is an example of how a fraud detection system in the real world works. This repository can help you simulate a system like that.**

## 2/ Model Development: ##

### a) Algorithm Selection

To tackle this fraud detection challenge, I turn to my trusted ally:

<p align="center">
  <strong>XGBoost, I choose you! ⚡</strong>
</p>

<p align="center">
  <img src="Images/gif.gif" alt="Funny GIF" />
</p>

**Why do I choose XGBoost?**
Simply because of its efficiency. Just imagine that a single false positive can result in a $10,000 loss for your customer.
Trusting such a task to a simple KNN model?
**Hear me out, bro. That should be illegal 🙃.**

I’m not saying KNN or Logistic Regression are bad algorithms, they just aren’t production-ready.
A system receives more and more data every day, while KNN tends to overfit on large datasets.
As for Logistic Regression, the model performs well only if the data satisfies certain assumptions, such as a linear relationship, which is very rare in real-world scenarios.

### b) Feature Engineering ###

This is the dataset I used for this project. You can download it from [here](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

And here are the features I used. I have classified them into three types:

**Tracking features:** 
I’m not sure if this is the right name. These features are used to track transactions and to generate other online features.  
They are not used for training the model. The tracking features include:

- **cc_num:** Unique identifier for each customer. 
- **trans_date_trans_time:** The time at which a user makes a transaction. In this simulation, this feature is automatically retrieved from the user’s system clock when they click **“Send Money”**
- **Merchant:** Who gets the money from the transaction.

**Offline Features:** The user provides the system with these features before making a transaction.



