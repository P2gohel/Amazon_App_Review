# Annotated Literature Review

> Interdisciplinary Group Project — Team 11
> *Improving Insight Quality in Amazon App Reviews through Sentiment Analysis and Anomaly Detection*

---

## Aggarwal, C.C. (2017)

**Aggarwal, C.C. (2017)** *Outlier Analysis.* 2nd edn. Cham: Springer. (Accessed: March 2026).

Aggarwal recommends that integrating several machine learning algorithms improves the effectiveness and robustness of anomaly detection, enhancing detection effectiveness and stability. This study supports the correlation analysis used in this project to validate that ensemble components contribute information rather than duplicating patterns.

The book also supports dimensionality reduction as a standard exploratory validation technique in unsupervised learning. It is used here to visually inspect whether fake reviews occupy different regions of the feature space, with significant mean differences confirming that the model captures interpretable patterns rather than random noise.

---

## Breunig et al. (2000)

**Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J. (2000)** 'LOF: Identifying Density-based Local Outliers', *SIGMOD '00: Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, pp. 93–104. [Accessed: 10 March 2026].

This study introduces the Local Outlier Factor (LOF) algorithm for detecting anomalies based on density deviation. Although the proposed project uses Isolation Forest, LOF provides theoretical background on unsupervised anomaly detection. The study is useful for understanding how abnormal behavioural patterns can be detected without labelled data, which aligns with the project's unsupervised fake review detection approach.

---

## Cadima and Jolliffe (2016)

**Cadima, J. and Jolliffe, I.T. (2016)** 'Principal Component Analysis: A Review and Recent Developments', *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 374, pp. 1–16. [Accessed: 1 March 2026].

Jolliffe and Cadima emphasise that PCA is widely used for exploratory validation in high-dimensional anomaly detection and analysis. The visual separation observed in this project supports the conclusion that the model captures meaningful structural differences rather than random variations.

---

## Chandola, Banerjee and Kumar (2009)

**Chandola, V., Banerjee, A. and Kumar, V. (2009)** 'Anomaly detection: A survey', *ACM Computing Surveys*, 41(3), pp. 1–58. [Accessed: 1 March 2026].

The survey by Chandola, Banerjee and Kumar provides an overview of anomaly detection techniques and explains that combining multiple detection approaches can improve robustness by distinguishing between normal and irregular behaviour.

---

## Dang and Zhang (2021)

**Dang, H. and Zhang, K. (2021)** 'Main Advantages and Disadvantages of Sentiment Lexicons: A Review', *Information*, 12(9), p. 368.

Dang and Zhang critically examine the strengths and limitations of sentiment lexicon-based approaches, noting their tendency to underperform supervised models in domain-specific settings due to their inability to capture contextual nuances such as negation and irony. This supports the comparative analysis conducted in this project, where VADER's rule-based approach struggled with the informal and context-dependent language found in Amazon app reviews, further justifying the adoption of a supervised machine learning approach as the primary classification method.

---

## Devlin et al. (2018)

**Devlin, J., Chang, M.W., Lee, K. and Toutanova, K. (2018)** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* [Accessed: 10 March 2026].

This landmark paper introduces BERT, a transformer-based language model that captures bidirectional context in text. The model significantly improved performance across NLP tasks including sentiment classification. The research supports the decision to compare traditional machine learning models with transformer-based approaches for improved contextual understanding in Amazon review analysis.

---

## Du et al. (2019)

**Du, J., Rong, J., Michalska, S., Wang, H. and Zhang, Y. (2019)** 'Feature selection for helpfulness prediction of online product reviews: an empirical study', *PLOS ONE*, 14(12), p. e0226902. Available at: <https://doi.org/10.1371/journal.pone.0226902> [Accessed: 23 February 2026].

This study explores how textual features influence the perceived helpfulness of online product reviews. The authors apply feature engineering and machine learning techniques to identify linguistic and structural attributes that impact review credibility. Their findings highlight the importance of review length, sentiment polarity, and content richness in shaping user trust. This research supports the proposed project's exploratory data analysis (EDA) phase, particularly in analysing review structure and behavioural patterns before model development.

---

## Emmott et al. (2013)

**Emmott, A., Das, S., Dietterich, T., Fern, A. and Wong, W. (2013)** 'A meta-analysis of the anomaly detection problem', *Proceedings of the 2013 SIAM International Conference on Data Mining*, pp. 920–928. [Accessed: 1 March 2026].

Emmott et al. recommend analysing score distribution and internal scoring behaviour as dynamic indicators where labels are unavailable. This supports the choice of validating model behaviour through anomaly score comparison instead of traditional supervised accuracy metrics, as seen in the analysis.

---

## Fawcett (2006)

**Fawcett, T. (2006)** 'An introduction to ROC analysis', *Pattern Recognition Letters*, 27(8), pp. 861–874.

Fawcett produced a comprehensive guide for ROC analysis, pointing out that ROC analysis assesses the discriminative capabilities of classifiers across all potential classification thresholds rather than just one. He illustrates that AUC is a better measure than accuracy because it isolates performance from the influence of class distribution or error cost in cases where class distribution is skewed. This is why ROC AUC is treated as a complementary measurement tool alongside accuracy and the F1-score for evaluating the effectiveness of this classification model. The Logistic Regression model achieved an AUC of **0.9678**.

---

## Hölzing et al. (2025)

**Hölzing, C.R., Meybohm, P., Happel, O., Kranke, P. and Meynhardt, C. (2025)** 'Transformer models enhance explainable risk categorization of incidents compared to TF-IDF baselines', *AI*, 6(9), p. 223. Available at: <https://www.mdpi.com/2673-2688/6/9/223> [Accessed: 22 April 2026].

This paper compares traditional TF-IDF-based models with transformer-based architectures such as Google's BERT. The study demonstrates that transformer models significantly improve contextual understanding and classification accuracy in complex textual data. This is particularly relevant to the sentiment classification phase of the project, where both traditional machine learning and modern NLP models are evaluated. The study justifies experimenting with contextual embeddings to improve classification reliability in noisy Amazon review data.

---

## Hutto and Gilbert (2014)

**Hutto, C.J. and Gilbert, E. (2014)** 'VADER: A parsimonious rule-based model for sentiment analysis of social media text', in *Proceedings of the International AAAI Conference on Web and Social Media*, pp. 216–225.

Hutto and Gilbert present VADER, a rule-based sentiment analysis tool designed for informal, short-form text. Using a validated lexicon and grammatical rules, it effectively captures sentiment intensity without requiring labelled training data. For this project, VADER provides a practical and efficient method for extracting sentiment from Amazon app reviews, making it a suitable choice for the sentiment analysis component of this study.

---

## Japkowicz and Stephen (2002)

**Japkowicz, N. and Stephen, S. (2002)** 'The class imbalance problem: A systematic study', *Intelligent Data Analysis*, 6(5), pp. 429–449.

Japkowicz and Stephen carry out an elaborate study on the impact of varying class distributions on the classification accuracy of conventional machine learning classifiers. They show that majority-class bias is a significant issue and suggest cost-sensitive methods, including the alteration of class weights, as a successful way of addressing the challenge without resorting to sampling techniques. In this study, the data contained approximately 62% positive reviews and 38% negative reviews — a slightly imbalanced dataset. To balance the classes, Logistic Regression was performed with balanced class weights, while Multinomial Naive Bayes was used with Laplace smoothing.

---

## Jindal and Liu (2008)

**Jindal, N. and Liu, B. (2008)** 'Opinion Spam and Analysis'.

This foundational study introduces the concept of opinion spam and categorises fake reviews into types such as deceptive reviews, brand-only reviews, and non-reviews. The authors propose behavioural and linguistic indicators of spam, including repetition patterns and unusual rating distributions. This work provides theoretical grounding for the anomaly detection phase of the proposed project, particularly in identifying sentiment–rating inconsistencies and repetitive patterns as signals of suspicious activity.

---

## Kohavi (1995)

**Kohavi, R. (1995)** 'A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection', *Proceedings of the 14th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 1137–1145.

Kohavi provides a rigorous examination of cross-validation and bootstrapping techniques, establishing stratified k-fold cross-validation as a reliable and recommended method for model evaluation. This directly informs the evaluation strategy of the present project, where 5-fold stratified cross-validation is applied to ensure robust and unbiased performance estimation. The cross-validation F1 of **0.9214** was nearly identical to the test F1 of **0.9227**, showing that the model generalises well with minimal risk of overfitting.

---

## Liu (2012)

**Liu, B. (2012)** *Sentiment Analysis and Opinion Mining.* San Rafael, CA: Morgan & Claypool Publishers.

Liu provides a solid foundation of sentiment analysis techniques, including lexicon-based methods and supervised learning models. The book outlines challenges such as sarcasm, informal language, and contextual ambiguity — issues highly relevant to Amazon app reviews. This source informs the methodological design of the sentiment classification phase and highlights the importance of preprocessing and feature engineering when dealing with unstructured consumer text.

Liu also acknowledges the practical challenges of messy, unstructured text data, a challenge that anyone applying these techniques to user-generated content has to negotiate. Liu provides the main theoretical foundation for this project, supplying the conceptual underpinning for applying sentiment analysis to Amazon app reviews in an organised and informed way.

---

## Liu, Ting and Zhou (2008)

**Liu, F.T., Ting, K.M. and Zhou, Z.-H. (2008)** 'Isolation Forest', in *Proceedings of the 2008 IEEE International Conference on Data Mining (ICDM)*, pp. 413–422.

This paper proposes the Isolation Forest algorithm, which isolates anomalies by randomly partitioning data points. It is computationally efficient and particularly suitable for high-dimensional datasets. This directly supports the project's choice of Isolation Forest to detect suspicious review behaviour, providing theoretical justification for selecting this algorithm over other anomaly detection techniques.

It also explains the realistic difficulty of unsupervised fake review detection: Isolation Forest assigns anomaly scores based on random partitioning, and the authors note that anomaly detection is probabilistic rather than deterministic — separation between normal and anomalous data is rarely perfect. This work supports the anomaly detection component of the project, providing a scalable and practical framework suitable for handling large and noisy datasets.

---

## Molnar (2019)

**Molnar, C. (2019)** *Interpretable Machine Learning.* Available at: <https://christophm.github.io/interpretable-ml-book/> [Accessed: 1 March 2026].

Molnar argues that interpretability is essential for trustworthy machine learning systems, especially in anomaly and fraud detection. Feature importance analysis emphasises trust and interpretability — considered essential in anomaly detection research — by explaining which variables influence the model's decisions. Therefore, evaluation done using feature importance strengthens the reliability of the detection work.

---

## Mukherjee et al. (2013)

**Mukherjee, A., Venkataraman, V., Liu, B. and Glance, N. (2013)** 'What Yelp Fake Review Filter Might Be Doing?', *Proceedings of the Seventh International AAAI Conference on Weblogs and Social Media*, 7(1), pp. 409–418. [Accessed: 5 March 2026].

Mukherjee et al. validate that fake review detection research relies on analysing linguistic and behavioural features to validate model outcomes. Their findings demonstrate that deceptive reviews show distinguishable textual and behavioural signs, which supports the use of descriptive statistical comparison as an evaluation method in this study.

---

## Ng and Jordan (2002)

**Ng, A.Y. and Jordan, M.I. (2002)** 'On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes', *Advances in Neural Information Processing Systems*, 14, pp. 841–848.

Ng and Jordan provide a theoretically motivated comparison of discriminative and generative classifiers, showing that when features are correlated — as is typical in text — discriminative classifiers such as Logistic Regression outperform generative classifiers such as Naive Bayes. The project's results are consistent with their theory: Logistic Regression achieved approximately 1.8% higher accuracy and 2% higher weighted F1 than Multinomial Naive Bayes.

---

## Pang, Lee and Vaithyanathan (2002)

**Pang, B., Lee, L. and Vaithyanathan, S. (2002)** 'Thumbs up? Sentiment classification using machine learning techniques', *Proceedings of the 2002 Conference on Empirical Methods in Natural Language Processing (EMNLP 2002)*, pp. 79–86.

Pang, Lee and Vaithyanathan use supervised machine learning methods to classify sentiment in product reviews and compare Naive Bayes, Maximum Entropy and Support Vector Machines on movie review data. The authors stress that model validity must be tested not only through quantitative measures such as accuracy, but also through qualitative analysis of the features used by the algorithm, ensuring the system actually learns linguistic characteristics of sentiment expression rather than noise patterns. This methodology is adopted in our study through examination of the top 20 weighted coefficients for both positive and negative sentiment alongside the confusion matrix.

---

## Pang and Lee (2008)

**Pang, B. and Lee, L. (2008)** 'Opinion Mining and Sentiment Analysis', *Foundations and Trends in Information Retrieval*, 2(1–2), pp. 1–135.

Pang and Lee present a solid axiomatic base for sentiment analysis, recommending the use of well-labelled datasets and traditional statistical machine learning algorithms such as Logistic Regression and Naive Bayes, which are known to perform reliably on similar classification problems. They also recommend excluding neutral instances from training data, arguing that focusing on strong sentiment signals produces cleaner and more accurate classification outcomes. This directly informs the present project, where reviews rated 1–2 stars are labelled negative and 4–5 stars positive, while 3-star reviews are excluded to improve classification clarity.

---

## Wang and Manning (2012)

**Wang, S. and Manning, C.D. (2012)** 'Baselines and Bigrams: Simple, Good Sentiment and Topic Classification', *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL 2012, Short Papers)*, pp. 90–94.

Wang and Manning find that bigram features, when used together with unigrams, provide a consistent improvement in sentiment classification across a variety of datasets. This validates the feature engineering done in the project: TF-IDF features are generated from both unigrams and bigrams, capped at 10,000 features. This is particularly useful for capturing sentiment in expressions like *"not good"* or *"very bad"*, which lose their sentiment when the words are isolated.

---

## Zhang (2004)

**Zhang, H. (2004)** 'The Optimality of Naive Bayes', *AAAI/IAAI*, pp. 562–567.

Zhang investigates the theoretical factors behind Naive Bayes's success, showing that the independence assumption does not prevent reliable results. The present study finds that the slightly inferior performance of Multinomial Naive Bayes (90.5% accuracy) compared to Logistic Regression validates its position as a strong baseline classifier rather than a failure of the algorithm itself.
