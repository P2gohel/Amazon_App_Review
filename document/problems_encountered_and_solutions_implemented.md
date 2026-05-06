# Problems Encountered and Solutions Implemented

> Interdisciplinary Group Project — Team 11
> *Improving Insight Quality in Amazon App Reviews through Sentiment Analysis and Anomaly Detection*

---

During the project, the team faced several issues related to coordination, time management, technical development, and model integration. These problems affected the progress of the work, but they also helped the team improve how tasks were planned, shared, and completed.

---

## 1. Team Coordination

At the beginning of the project, communication between team members was not always well structured. Some discussions were held informally, which caused confusion about who was responsible for certain tasks. In some cases, two members worked on similar areas, while other tasks were delayed because dependencies had not been clearly discussed.

This became more noticeable when the team had to decide which features should be included in the dashboard. Different members had different ideas, and without a clear decision-making process, the scope of the dashboard became unclear.

**Solution.** To solve this, the team started holding regular weekly meetings and recording meeting minutes. Tasks were discussed more clearly before each sprint, and responsibilities were assigned based on each member's role and skills. When there were disagreements, the team discussed the options and made decisions together.

This improved coordination because everyone had a clearer understanding of their tasks and deadlines. It also helped reduce repeated work and made the team more organised during the later stages of the project.

---

## 2. Time Management

Time management was another challenge because team members also had other coursework and deadlines. The preprocessing stage took longer than expected because the Amazon review dataset needed more cleaning than originally planned. Missing values, duplicate reviews, inconsistent text, and formatting issues had to be handled before EDA and modelling could be done properly.

Because preprocessing took extra time, later tasks such as model testing, visualisation, and dashboard development had less time available. This showed that the original timeline had been too optimistic.

**Solution.** The team reviewed the Gantt chart and adjusted the remaining tasks based on priority. More time was given to important stages such as anomaly detection, sentiment analysis, and evaluation. Less essential features were moved to later stages or simplified.

From this, the team learned that project planning needs to include extra time for unexpected issues, especially when working with real-world datasets. The team also learned that Agile planning should be flexible, and changes should be made early rather than waiting until the project falls behind.

---

## 3. Technical Challenges

The project also presented several technical challenges.

### 3.1 Dataset size and runtime

The full dataset was large and slow to process on personal laptops. Running repeated cleaning, EDA, and modelling steps took time and sometimes made development inefficient.

**Solution.** The team improved the code by using more efficient processing methods where possible. The cleaned dataset was saved after preprocessing so that the same cleaning steps did not need to be repeated every time.

### 3.2 Tuning the Isolation Forest

Tuning the Isolation Forest model for fake review detection was non-trivial. At first, some settings flagged too many reviews as suspicious, while other settings missed too many unusual patterns. This made it difficult to decide which threshold was most suitable.

**Solution.** The team tested different contamination levels, including 5%, 10%, and 15%. The **10%** threshold was selected because it gave a balanced result. The team also added rule-based detection and ensemble voting so that suspicious reviews were not judged by one method alone.

### 3.3 Class imbalance in sentiment analysis

The sentiment analysis model also had a class imbalance issue because the number of positive and negative reviews was not equal. This could have caused the model to perform better on the majority class.

**Solution.** The team used stratified train/test splitting, 5-fold cross-validation, F1-score for evaluation, and balanced class weights for Logistic Regression.

### 3.4 Environment and dependency issues

The team also faced dependency issues because members had different Python versions and library versions. This caused some code to work on one laptop but not another.

**Solution.** The team created a `requirements.txt` file and agreed to use consistent library versions.

Overall, the technical challenges helped the team understand the importance of clean preprocessing, proper model tuning, and shared development environments.

---

## 4. Dashboard Integration

The dashboard was another area where the team faced difficulty. The models and visualisations worked separately in notebooks, but connecting them into one dashboard required additional testing. Issues appeared when handling user input, displaying results clearly, and making sure the dashboard output matched the model predictions.

**Solution.** The team tested the dashboard in smaller parts first. The sentiment model, fake review detection logic, and visual outputs were checked separately before being combined. This helped identify errors more easily and made the final dashboard more stable.

Even when the model works correctly in Python notebooks, extra testing is needed when turning it into an interactive prototype.

---

## 5. Overall Reflection

These problems showed that the project required more than just technical development. The team had to improve communication, planning, task sharing, and testing throughout the process. The main improvement was that the team became more organised after the first sprint by using meetings, Trello, GitLab, and clearer role allocation.

The biggest lesson was that real-world data projects are difficult to plan perfectly from the start. Data cleaning, model tuning, and integration often take longer than expected. However, by adjusting the plan and working collaboratively, the team was able to complete the main objectives of the project.
