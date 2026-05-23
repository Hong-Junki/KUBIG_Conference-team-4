# Breaking the trend: Anomaly detection models for early warning of socio-political unrest

**Luca Macis a,\*, Marco Tagliapietra a, Rosa Meo b, Paola Pisano a**

- a: Department of Economics and Statistics Cognetti de Martiis, University of Turin, Italy
- b: Department of Computer Science, University of Turin, Italy

> Technological Forecasting & Social Change 206 (2024) 123495
> Received 6 March 2023; Revised 4 May 2024; Accepted 4 June 2024
> Open access: CC BY license

**Keywords:** Anomaly detection, Conflict prediction, Early warning system, Autoencoder, Artificial intelligence

---

## Abstract

This paper presents an innovative Early Warning System for predicting conflicts and unrest based on Anomaly Detection, identifying sudden and unexpected changes in behavioral patterns that may indicate the potential for these events to occur. This approach draws inspiration from various fields – including industry, such as manufacturing, physics and networking – but its application in the domain of diplomacy is entirely new. The system, tested on three case studies, showcase its ability to enhance open-source intelligence technique in the diplomatic arena. The study provides a fresh perspective on predictive analytics and focuses on examining outbreaks.

---

## 1. Introduction

Conflict prevention is critical to international relations and diplomacy, as early intervention can often lead to conflicts being resolved before they are able to escalate into larger and more destructive events. Early warning systems play a crucial role in this context as they identify potential risks and threats and provide decision-makers with timely information to help define policies for mitigating or preventing any conflict.

Early warning systems are often based on data analysis and advanced computational techniques thanks to the large amounts of data gathered in information systems, shared via the Internet, and advancements in the fields of Machine Learning and Artificial Intelligence. Data can be extrapolated from various sources, including open-sources, satellite imagery, and social media. However, knowledge discovered from data is the final step in a much longer process that begins with identifying patterns and anomalies that may indicate a potential conflict. This allows decision-makers to address the underlying causes of conflict proactively and to take action to prevent its emergence.

Prior to these technological advancements taking place, potential conflict was traditionally identified largely by way of individual diplomatic and political knowledge, intuition, and subjective judgment. Although this approach had some success, intuitive approaches to conflict prediction have been criticized for their lack of rigor and strong subjectivity. Human analysts are often biased due to their own experiences, beliefs, and perspectives, leading to inconsistent and unreliable predictions (Gleditsch, 2002), thus entailing a more reactive than a proactive approach. Data and technology have the potential to revolutionize the field of conflict prevention, providing early warning signals of potential unrest. With the advent of big data from multiple sources and machine learning, vast amounts of data can now be analyzed in real time, identifying potential risk factors before they escalate into full-blown conflicts, and standardizing predictions in an objective and accountable way. This can improve the accuracy and timeliness of conflict prevention efforts and give decision-makers the technical support they require in order to act promptly and effectively at the first sign of crises.

Governments and international organizations now have come to acknowledge the importance of using technology and data analysis for conflict prediction in order to prevent or mitigate the imminent risk. This has been the case for several decades, as demonstrated by early studies on preventive diplomacy (George, 1999; Ackermann, 2003) and more recent works, such as the book *Violence, Wars, Peace, Security* (Wallensteen et al., 2018). The use of data and technology may potentially facilitate early peacekeeping interventions, which are a crucial factor in reducing conflict, as indicated in the research by Håvard Hegre, Lisa Hultman, and Håvard Mokleiv Nygård (Hegre et al., 2019b). Their study simulated a scenario in which the United Nations ceased its peacekeeping efforts after 2001. They found that, compared to the actual number of countries with ongoing peacekeeping efforts, three to four more countries would have suffered major conflicts by 2013 in the absence of the UN's efforts.

In response to this, various organizations initiated the development of data analysis and data collection and interpretation systems, such as the Global Conflict Risk Index (Halkia et al., 2020a) by the Joint Research Center of the European Commission and the Violence and Impacts Early-Warning System (VIEWS) (Hegre et al., 2019a) developed by Uppsala University.

This paper outlines the development and outcomes of a new Early Warning System, using Machine Learning techniques. More specifically, it uses an Anomaly Detection approach for the purposes of preventing conflict and unrest. It was chosen to use this approach, which has never been used in this field, and which differs from previous supervised methods, as it is more apt to identify anomalies in a country. The promising results of the three case studies infer that this approach could be used for a wider range of conflicts. The Section 2 and the Section 4 of this paper provide a comprehensive overview of current conflict prevention models and illustrate how paper models could provide a valid alternative to those already in use.

---

## 2. Artwork (Related Work)

The conflict prediction field has traditionally used various Machine Learning models, including Logistic Regression (Halkia et al., 2020a), Dynamic Multinomial Logit (Hegre et al., 2012), Random Forest (Muchlinski et al., 2016), naive Bayes classifiers (Perry, 2013), and Neural Networks (Beck et al., 2000). All of these methods are supervised, meaning that they require labeled data, with domain experts providing the correct output for each input. This allows the algorithms to learn patterns and to make predictions based on past data. While supervised methods rely on labeled data to learn patterns and make predictions, unsupervised methods uncover underlying structures in unlabeled data, making them effective for anomaly detection where supervised methods might fall short.

This work, however, adopts an innovative approach based upon Anomaly Detection, which can be treated as an unsupervised method. Unsupervised methods do not require labeled data and instead identify patterns or anomalies within the data themselves. Anomaly Detection has achieved successful results in several domains, such as industry (Siegel, 2020) and physics (Nakazawa and Kulkarni, 2019), as well as with multi-variate time series (Zhang et al., 2021). It is an attractive method of conflict prediction as it does not require domain-expert intervention and it is well-suited to detecting rare events. This is discussed in the book *Understanding Deep Learning* (Ranjan, 2020), which argues that anomaly detection can identify unusual patterns in data that may indicate a rare occurrence, whereas supervised methods struggle to manage rare events due to their reliance on statistical metrics which are better suited to identifying frequent categories and the majority class. Ranjan's book focuses on using an Autoencoder model in Anomaly Detection for rare-event detection.

An Autoencoder is an unsupervised machine learning model which aims to identify patterns or anomalies in data. The first step is to divide the data into two groups: the majority class, which represents the normal state of the data, and the minority class, which represents the rare events. The Autoencoder then compresses and extracts the important information from the majority class, reconstructs it and creates a latent representation (hereinafter, Latent Space) of the majority class. During the last inference step, the Autoencoder compares the reconstructed data with the original data. Once the Latent Space is built, new data are given to the Autoencoder as input, with a view to reconstructing that input. If there is little difference between the new original data and the new reconstructed data, this means that the sample belongs to the majority class, or the normal state. However, if there is a large difference between the two, this suggests that the sample belongs to the minority class and is thus an anomaly.

In the paper entitled *A Variational Autoencoder Solution for Road Traffic Forecasting Systems* (Boquet et al., 2020), the author shows that anomaly detection utilizing an autoencoder can be a valuable approach to predicting rare events, such as sudden shifts in road traffic patterns; it does so by learning a subspace that captures the underlying characteristics of the traffic data. Boquet proposes an autoencoder model that can extract valuable, compressed information from complex and multidimensional road traffic data sets and assign missing values online, without supervision. This extraction and compression technique improves the accuracy of forecasting systems. The author also highlights that the Latent Space, i.e. the compressed information learned by the model, is suitable for traffic anomaly detection. Similarly, events such as armed conflicts, coups, and unrest can be considered anomalies in a country's history and can be predicted as rare events. In the context of this academic evidence, this paper uses an unsupervised Anomaly Detection Autoencoder model for a conflict prevention Early Warning System. The Sections 3 and 7 below will discuss the databases and the results of this approach.

---

## 3. Dataset implementation

Dataset selection is a critical factor in a heavily data-driven model. The datasets adopted for conflict prediction primarily fall into two categories: social and diplomatic.

Many forecasting works have relied on data obtained from social media platforms, particularly Twitter, due to its ease of access. In recent years, numerous examples of conflict prediction using social media have emerged, such as the work by Korkmaz (Korkmaz et al., 2015). That article focuses on the use of multiple data sources, including Twitter, to predict and analyze civil unrest in six Latin American countries, highlighting the importance of using this platform as a dataset in predicting civil unrest. Another example of conflict prediction using Twitter is demonstrated in Ryan Compton's paper *Using Publicly Visible Social Media to Build Detailed Forecasts of Civil Unrest* (Compton et al., 2014). Compton presents a data mining system that generates forecasts of civil unrest incidents in Latin America using public posts on Twitter and Tumblr. The system identifies informational posts through filters applied to tens of millions of posts per day and generates predictions by annotating filtered posts with demographic, spatial, and temporal information.

However, as noted in the paper *Towards Understanding the Use of Telegram by Political Groups in Brazil* (Junior et al., 2021), Twitter is no longer used as a platform for organizing insurrections due to the prohibition imposed on violent threat tweets and on the tracking and persecution of Twitter users by authoritarian regimes. Consequently, a new social messaging platform, Telegram, has taken its place. This study specifically examines the use of Telegram by political communities in Brazil, with the results revealing a significant increase in political mobilization on this platform in recent years. Telegram's large-group structure has proven to be more effective in spreading messages compared to other preceding social media channels.

Telegram, as mentioned in the paper *Telegram: Data Collection, Opportunities and Challenges* (Khaund et al., 2021), remains an unexplored platform for data mining purposes, with only a proposed framework for conducting data mining and no notable applications in the field of conflict prediction.

For these reasons, we discarded social datasets and focused on diplomatic datasets. However, in creating our dataset, we decided to use time-series datasets that were updated regularly and contained daily observations, leading us to consider the Armed Conflict Location and Event Data Project (ACLED) (Raleigh et al., 2023) and Global Database of Events, Language, and Tone (GDELT) datasets and to discard other diplomatic, economic, and sociopolitical sources, such as the World Bank – which is adopted in several conflict prevention projects, including the Global Conflict Risk Index (Halkia et al., 2020b) – due to its non-daily variables.

The use of both datasets has been demonstrated to be valuable in conflict prediction research, as highlighted by other works (Matina et al., 2019), as this provides a more comprehensive and diverse set of variables for the model: ACLED offers a more exhaustive view of events such as conflicts, protests, and riots, while GDELT focuses more on Natural Language Processing and sentiment analysis. This combination of perspectives improves the performance of the analysis models compared to the use of either dataset alone.

In addition, this study benefits from both human-coded and machine-coded diplomatic databases in order to overcome the limitations of relying on a single dataset. The human-coded database, ACLED, offers the advantage of accurate information, but with a weekly update frequency. Conversely, the machine-coded database, GDELT, provides a more frequent update cycle – every 15 min – including news from various media sources in over 100 languages, dating back to 1979 (Leetaru and Schrodt, 2013). The merits of merging and combining these two datasets were explored in the paper *Comparison Metrics for Large Scale Political Event Data Sets* (Schrodt and Analytics, 2015).

Data wrangling had to be performed in order to obtain clear and functional variables from the two datasets. The ACLED data are available for download on its platform (ACLED, 2022b) in the form of .csv files, which can be filtered by date range and event type. The types of events recorded in the dataset include battles, explosions and remote violence, violence against civilians, riots, protests, and strategic developments.

One limitation of the ACLED dataset is that its coverage period varies by country and region. For example, while most African regions have data from 1997, some European and Asian regions have data from 2018. An exhaustive country and time coverage list is available on the ACLED website (ACLED, 2022a).

To wrangle the dataset, we initially downloaded separate ones for each event type and then created new copies for each dataset, grouping them by ISO 3166-1 alpha-3 code and date. ISO 3166-1 alpha-3 codes are three-letter codes used to identify countries in the ISO standard. This produced a dataset in which each row corresponded to a day in a specific country, with the number of events of a specific type that occurred on that day in that country.

The GDELT dataset, on the other hand, contains many articles, as transpires from its Power Law distribution of the number of mentions per article (as shown in Fig. 1). However, many of these articles are considered noise, lacking significant information relevant to our analysis. To overcome this, the dataset was reduced, retaining only articles that received at least 10 mentions within the first 15 min of publication. This reduction strategy resulted in a dataset size of just 1.4 % of the original. The threshold of 10 mentions was determined using the elbow method, as depicted in Fig. 1. This reduction strategy may lead to the exclusion of important information from countries where freedom of the press is suppressed and international media attention is limited; however, this approach was deemed acceptable as there are relatively few countries of this nature. The Dynamic GCRI – a conflict risk model developed by the European Commission's Joint Research Centre which uses event data to predict conflict – adopted the same approach (Matina et al., 2019) based on the idea that, when an event actually happens, it will be reproduced by more than one media source and in more than one article. Moreover, the Dynamic GCRI highlights that the inclusion of all available information within the GDELT database would lead to greater bias than the exclusion of some information.

Although the results obtained were promising, for the purpose of global implementation, a careful assessment of the freedom of press in each country is recommended by monitoring relevant indices and tools. To this end, we suggest using the *Media Freedom Analyzer* proposed by Laura Schneider (Schneider, 2019) or other established indices such as the *Freedom of the Press Index* by Freedom House and the *Press Freedom Index* by Reporters Without Borders, analyzed by Ewa Sapiezynska and Claudia Lagos (Sapiezynska and Lagos, 2016).

> **[Fig. 1]** GDELT mentions distribution from July 1, 2022 to August 1, 2022. P(number of mentions > k) follows a Power Law distribution. k = 10, P(number of mentions > 10) ≈ 1.4%.
>
> **[그림 내용]** X축: Number of mentions (0~120), Y축: Number of articles (0~800k). 전형적인 멱법칙(Power Law) 분포 — 대부분의 기사가 극소수 멘션(10 미만)을 받으며 긴 꼬리(long tail)가 오른쪽으로 뻗어있음. k=10 지점에 점선(threshold k)이 표시되어 있고, 이 임계값 이상인 기사는 전체의 약 1.4%에 불과. 이를 기준으로 노이즈를 제거했음.

The GDELT dataset includes 62 attributes, which are grouped into the following four categories:

- **Date attributes**, used to identify the globally unique identifier number and the date of the event;
- **Actor attributes**, used to identify the countries actively or passively involved in the event, using a unique 3-letter ISO 3166-1 alpha-3 code. We use this attribute to identify and associate the observation with a specific country. In the case of the absence of this attribute, the "Event geography" is applied.
- **Event geography**, used to identify geographical information in cases where the actor attributes are null.
- **Event action attributes**, which include the Average Tone – a sentiment value obtained through sentiment analysis – the Number of Mentions of the event received in the last 15 min, and the QuadClass taxonomy. The latter classifies the event type into four primary categories: verbal cooperation, material cooperation, verbal conflict, and material conflict. Finally, there is the Goldstein Scale, a numeric score between −10 and 10 which captures the event's theoretical potential impact on a country's stability.

Detailed information about the attributes can be found in their documentation (GDELT, 2022).

It is important to note that the GDELT dataset is obtained by semantically analyzing each news item using Natural Language Processing. This analysis identifies the event location, the actors involved, and the other relevant elements embedded in the news. Actors are associated with ISO 3166-1 alpha-3 codes. News articles lacking information on actors or event locations were removed.

The resulting dataset was then grouped by country and date, as was done for the ACLED dataset. Only the ISO 3166-1 alpha-3 code associated with the actors actively or passively involved was recorded for each news item, excluding other actors. The country was recorded based on the event location in cases where the news item did not report actors. The dataset included additional attributes, such as the number of news items, the GoldsteinScale, and the Average Tone associated with each newsgroup belonging to the same QuadClass, for each date and country. In addition to the variables extracted and grouped daily from the two databases, a new variable called Violence was created, inspired by the definition of violence given by ACLED (ACLED, 2022c):

> Violence rates are calculated using the number of battle events, explosion or remote violence events, violence against civilians, and riot events, as well as the excessive force against protesters, sub-event type of the protest event type, excluding peaceful protest events and protest-with-intervention events.

The Violence variable was used as the target variable for the final case study to predict the onset of violence in Sri Lanka on 9 May 2022. In the final case study, the system attempted to predict the sudden increase in unrest events considered in the construction of the variable itself.

---

## 4. Methodology

In order to implement Anomaly Detection – a key method for recognizing unusual occurrences within data – we utilized an Autoencoder constructed utilizing an Encoder-Decoder architecture.

The Encoder-Decoder architecture is based on the idea of compressing data into a lower-dimensional representation, known as Latent Space, and then reconstructing the original data from this compressed representation. The Encoder component processes the input data matrix X and compresses it into a lower-dimensional representation. The Decoder component is responsible for reconstructing the original data from the compressed representation.

This compressed representation H aims to capture the most salient features of the original data and acts as input for the Decoder component, which is responsible for reconstructing the original data (Eq. (1)).

```
f_enc(X) = H                    ... (1)
```

The decoder component uses the compressed representation, H, as input and reconstructs a new matrix, X̃, which aims to be as similar as possible to the original matrix X (Eq. (2)).

```
f_dec(H) = X̃                   ... (2)
```

The squared error SE (Eq. (3)) is used as a similarity measure to calculate the difference between the original data and the reconstructed data for each day. Specifically, it computes the dissimilarity between the original data xᵢ, and the reconstructed data x̃ᵢ, for the i-th day, where i is the index for the day. This provides a scalar value representing the dissimilarity between the original data and the reconstructed data for each day. By minimizing the SE, the decoder aims to reconstruct the original data as accurately as possible for each day, while the encoder's goal is to identify the most compact data representation for each day.

```
SE(xᵢ, x̃ᵢ) = (xᵢ − x̃ᵢ)²    ... (3)
```

> **[Fig. 2]** Encoder-Decoder architecture diagram.
>
> **[그림 내용]** 왼쪽에서 오른쪽으로 흐르는 구조도. 입력 행렬 X(넓은 직사각형) → Encoder(f_enc) → 압축된 잠재 표현 H(좁은 직사각형, Latent Space) → Decoder(f_dec) → 재구성 행렬 X̃(넓은 직사각형). X에서 H로 갈수록 차원이 줄어들고, H에서 X̃로 갈수록 다시 원래 차원으로 복원되는 모래시계(bottleneck) 형태.

Both models were trained with data from "normal" days (i.e. days far away from a conflict) to establish a baseline, or in other words, a Latent Space for what is considered normal behavior. During validation and testing, days that deviated significantly from this normal behavior were labeled as anomalous. These anomalies are indicative of potential "pre-unrest" days, which are days leading up to an outbreak of civil unrest or conflict.

To determine the "normal" and "pre-unrest", the data was first filtered to remove actual "unrest" days, namely the days with the highest number of violent occurrences and unrest reported in news sources. The remaining data were then analyzed to identify patterns and to classify the days into "normal" and "pre-unrest" categories. The purpose of this process was not to identify "unrest" days, being a trivial task, but rather to distinguish between "normal" days and "pre-unrest" days, which may be characterized by increased tension, signs of social unrest, or other indicators of potential violence.

Two parameters were used to make the distinction between "normal" days and "pre-unrest" days through statistical analysis: **Target Variable Threshold** and **Pre-Unrest Days**. Unlike previous studies which used a fixed threshold to define the state of war and unrest (Halkia et al., 2020a) and a set number of days to predict unrest in the short term, such as, 30 or 90 days (Rustad et al., 2011; Hegre et al., 2019a, 2019b), this study used an optimization system which selects the threshold of "unrest" and the number of "pre-unrest" days; this is unique to each country and has a predefined range.

The **Target Variable Threshold** refers to the value which, if surpassed, defines an event as being in a state of "unrest". The **Pre-Unrest Days** are the n days prior to the "unrest" day. These parameters were determined through a Grid Search optimization technique. The Grid Search was used to distinguish effectively between the "normal" days and the "pre-unrest" days by identifying the optimal values for the Pre-Unrest Days and Target Variable Threshold from a set of possible values. To define an "unrest" day, the value of the target variable – i.e., the variable examined to assess the sociopolitical unrest in a given country – had to be higher than at least than 98 % of the values of the same target variable on all the other days in the country dataset in question, as it had to be a rare event.

The possible values for each case study, which will be analyzed in detail in Section 6, are listed in Table 1.

**Table 1.** Case study parameters. The start date is the first day of coverage by ACLED; the end date is the day of unrest. (\*) 98% of days have a value below the lower threshold; (\*\*) is the highest value of the target variable.

| Case study | Date range | Target variable | Threshold | Pre-unrest days |
|---|---|---|---|---|
| Ukraine | Jan. 01, 2018 – Feb. 24, 2022 | Battle fatalities | 5\*–52\*\* | 30–90 |
| Burkina Faso | Jan. 01, 1997 – Jan. 24, 2022 | Battle fatalities | 6\*–147\*\* | 30–90 |
| Sri Lanka | Jan. 01, 2010 – May 9, 2022 | Violence | 2\*–18\*\* | 30–90 |

The selection of pre-unrest days was established within a range of one to three months with the aim of assessing the short-term efficacy of the approach. The lower limit was specifically set at 30 days, representing the minimum prediction threshold reported in prior literature, as mentioned above. For each combination of possible values (i.e., Threshold - Pre-Unrest Days), the model's performance in separating the "pre-unrest" days from the "normal" days was evaluated during validation. The combinations that performed best during validation are listed in Table 2.

**Table 2.** Optimal parameters determined during validation.

| Case study | Target variable | Threshold | Pre-unrest days |
|---|---|---|---|
| Ukraine | Battle fatalities | 6 (98.1 %)^a | 56 |
| Burkina Faso | Battle fatalities | 52 (99.8 %)^b | 63 |
| Sri Lanka | Violence | 15 (99.8 %)^c | 47 |

> ^a 98.1% of days in the Ukraine dataset have battle fatalities lower than 6.
> ^b 99.8% of days in the Burkina Faso dataset have battle fatalities lower than 52.
> ^c 99.8% of days in the Sri Lanka dataset have violence lower than 15.

The values reported in Table 2 were those ultimately used during testing to evaluate the performance for each case study. A more comprehensive examination of the determination of the parameters and the framework of the data is provided in the Appendix.

The Early Warning System implemented uses two different models based on the Encoder-Decoder architecture: an **Encoder-Decoder Principal Component Analysis (PCA)** model and a **Long Short-Term Memory (LSTM) Autoencoder**. The Encoder-Decoder PCA model was selected for its computational efficiency, as it can optimize two key parameters – Target Variable Threshold and Pre-Unrest Days – quicker than the LSTM Autoencoder. This is due to the fact that the Encoder-Decoder PCA model uses only linear combinations when identifying the Latent Space. Using the computer provided for this research, the Encoder-Decoder PCA model takes approximately 0.01 s to compile, compared to the 2 min taken by the LSTM Autoencoder. However, the LSTM Autoencoder can use non-linear functions, leading to a more accurate representation of the Latent Space and ultimately a better performance. Accordingly, the Encoder-Decoder PCA model was used to identify rapidly the optimal parameters, while the LSTM Autoencoder used those parameters to achieve a better performance.

---

## 5. Performance measure

Two plots were used to evaluate the performance of the proposed Early Warning System's performance: the **reconstruction error plot** and the **receiver operating characteristic (ROC) plot**. The reconstruction error plot displays the chronological progression of test set days from left to right, up until the day before the unrest event, as shown in Fig. 3.

Each day is represented as a dot, with the height of the dot indicating the squared error (SE) value. The higher the dot, the higher the SE value and, therefore, the more likely the day is considered a "pre-unrest" day. Orange circles represent "normal" days, while coral crosses represent "pre-unrest" days. A horizontal black line shows the threshold for separating "normal" days from "pre-unrest" days.

> **[Fig. 3]** Reconstruction error plot of the ideal model.
>
> **[그림 내용]** X축: 시간(왼→오른쪽), Y축: SE(재구성 오차). 왼쪽 영역에 주황색 원(○, "normal" days)이 낮은 SE로 밀집, 오른쪽 unrest 이벤트 직전 영역에 산호색 십자(×, "pre-unrest" days)가 높은 SE로 밀집. 수평 검은 실선이 threshold로 두 그룹을 깔끔하게 분리. 파란색 이동평균선(최근 15일)이 unrest 접근 시 threshold를 넘어 상승하는 패턴. 이상적인 모델에서는 threshold 아래=normal, 위=pre-unrest로 완벽 분리됨.

A ROC plot was also used to assess further the model's ability to distinguish between the two types of days. The ROC plot displays the Area Under the Curve (AUC), which is used as a precision measure to evaluate the model. AUC values close to 1 (100 %) indicate that the model has distinguished all days perfectly, while an AUC value of 0.5 (50 %) indicates that the model cannot distinguish between the two types of days. The ROC plot examines the relationship between the True Positive Rate (TPR, fraction of true positives) and False Positive Rate (FPR, fraction of false positives) or the relationship between true and false alarms. The closer the AUC in the ROC plot is to 1, the more effective the model is in separating true and false alarms. Therefore, the AUC can be interpreted as the probability of a day being a true alarm (a "pre-unrest" day) if the threshold is crossed, or the probability of being a "normal" day if the threshold is not crossed. This evaluates more accurately the model's ability to distinguish between "normal" days and "pre-unrest" days and, thus, its effectiveness in predicting and preventing potential conflicts.

---

## 6. Case studies

In order to evaluate the effectiveness of the models, recent case studies concerning socio-political unrest events were analyzed. For each case, the ACLED database was examined to identify any variables that underwent a dramatic change in trend leading up to the event. The case studies chosen for this analysis were carefully selected to represent a range of potential scenarios in which the Early Warning System may be used.

### 6.1. Russian invasion of Ukraine

The first case study analyzes a critical event in the continuing Russia-Ukraine conflict that escalated significantly on February 24, 2022. This date marked a dramatic increase in battle-related fatalities, originating from military tensions that began in February 2014. The main objective of this case study is to explore whether this sharp increase in fatalities could have been anticipated earlier through the early warning system.

In the 56 days before the invasion, a total of 1600 events were recorded in Ukraine leading up to the Russian invasion. Despite this, fatalities remained relatively low during this period, with only 31 deaths registered prior to February 24, in stark contrast to the 103 fatalities recorded on the day of the invasion itself. 92 % of the reported events were categorized as Explosions/Remote violence and Battles, indicating prevalent but initially low-lethality conflict activities. Additionally, the geographical distribution of these events was strikingly concentrated, with >92 % occurring in two eastern regions near the Russian border – Donetsk and Luhansk (Fig. 4).

> **[Fig. 4]** Evolution of battle fatalities (red) and battles (blue) in Ukraine from Jun. 24, 2021, to Apr. 24, 2022. Source: ACLED.
>
> **[그림 내용]** X축: 2021-07 ~ 2022-04, Y축: 0~300. 파란 선(battles)은 2021년 하반기~2022년 초 꾸준히 낮은 수준(일 5~20건) 유지. 빨간 선(battle_fatalities)은 침공 전까지 거의 0 근처. **2022-02-24(검은 점선, 침공일)**에 fatalities가 갑자기 100+로 폭등, 이후 200~300 수준 급등. 핵심 관찰: 침공 전 전투 건수는 있었지만 사상자는 극히 낮았음 → 침공일 급변.

### 6.2. Burkina Faso coup

The second case study concerns the overthrowing of the democratically-elected government headed by Roch Marc Christian Kaboré on behalf of a fringe army, which occurred between January 23 and 24, 2022. The study primarily explores whether the spike in battle-related fatalities observed just days prior to January 24 could have been anticipated by an early warning system, despite unclear trend changes.

In the 58 days preceding the critical increase in fatalities on January 19, a total of 437 events were reported, leading to 615 fatalities. However, the trend in fatalities did not show a noticeable increase until the five days before the coup. During this period, 41 events occurred with 204 fatalities — an alarming rise in deaths. Most of these events were battles and strategic developments, but there was also significant violence against civilians, riots, and other violent activities. Moreover, from November 22, 2021, to January 24, 2022, the militants involved in the coup engaged in significant property destruction, including sabotage of telecommunications systems, burning buildings and government infrastructures, vandalism, and destruction of schools. These acts of sabotage and destruction underscored the intense escalation leading up to the coup (Fig. 5).

> **[Fig. 5]** Evolution of battles (blue) and battle fatalities (red) in Burkina Faso from Jan. 01, 2021, to Feb. 28, 2022. Source: ACLED.
>
> **[그림 내용]** X축: 2021-01 ~ 2022-01, Y축: 0~140. 빨간 선(battle_fatalities)이 2021년 내내 산발적 스파이크(20~80 수준)를 보이다가, 2021-09경 최대 140+ 피크. 파란 선(battles)도 비슷한 패턴. 초록 점선(2022-01-19): 쿠데타 직전 fatalities 피크(107명). 검은 점선(2022-01-23): 쿠데타 발생일. 핵심 관찰: 쿠데타 5일 전부터 급격한 사상자 증가, 그 전에는 불규칙하지만 점진적 상승 추세.

### 6.3. Sri Lanka's protests

The last case study focuses on the violence variable, derived from the ACLED dataset's disorder factors. Beginning with peaceful protests in late March 2022, the situation quickly escalated after declaring a state of emergency on April 1, witnessing a surge in mass protests that progressively became more violent. Notably, by May 11, the intensity of the unrest compelled the government to deploy the army with shoot-on-sight orders. This study notably investigates if the significant rise in violence from May 9, 2022, was foreseeable. Prior to this peak, several incidents hinted at the escalating tension, starting with inter-group clashes and worsening with responses to the economic downturn, fuel shortages, and the demanding of President Gotabaya Rajapaksa's resignation, culminating in nationwide strikes on April 28. The increasing frequency and intensity of these events painted a clear picture of the unrest leading up to the mentioned surge in violence (Fig. 6).

> **[Fig. 6]** Evolution of violence (red) and protests (light blue) in Sri Lanka from Jan. 01, 2022, to Jun. 01, 2022. Source: ACLED.
>
> **[그림 내용]** X축: 2022-01 ~ 2022-06, Y축: 0~80. 하늘색 선(protests)이 3월 말부터 급증하여 4월 중 최대 60~80건/일. 빨간 선(violence)은 3월까지 거의 0이다가 4월부터 소폭 증가, **5월 9일경 급등**(일 14건). 검은 점선(2022-04-01): 비상사태 선포. 검은 점선(2022-05-11): 군 투입(사살 명령). 핵심 관찰: 평화 시위 → 비상사태 → 폭력 전환 → 군 투입의 에스컬레이션 패턴이 명확.

---

## 7. Results

The aim of the first case study was to detect the 56 "pre-unrest" days prior the Russian invasion of Ukraine in 2022 using an LSTM Autoencoder. As already mentioned in the Section 4, the specific parameters used in the study are listed in Table 3.

**Table 3.** Ukraine parameters.

| Battle fatalities threshold | Pre-unrest days |
|---|---|
| 6 (98.1 %)^a | 56 |

> ^a 98.1% of days in the entire dataset have battle fatalities < 6.

As noted, the optimal values for the parameters Battle Fatalities Threshold and Pre-Unrest Days were determined through a Grid Search optimization technique. The case study evaluated the model's ability to detect the 56 "pre-unrest" days preceding the Russian invasion of Ukraine on February 24, 2022. The model was considered successful if it could classify a day as "pre-unrest" even if the number of battle fatalities on that specific day was < 6; the requirement was that this day was one of the 56 days leading up to an "unrest" day with > 6 battle fatalities.

The analysis of the reconstruction error plot (as shown in Fig. 7) reveals that the days preceding the Russian invasion of Ukraine (February 24, 2022) exhibit a higher reconstruction error than the days deemed normal (represented by orange circles in the plot). The threshold for separating "normal" days from "pre-unrest" days, determined during the validation phase in the framework mentioned above, is represented by the black line in the plot. The blue line represents a moving average concerning the reconstruction error of the last 15 days. This moving average was added as an indicator of the trend in the reconstruction error. The day the moving average surpasses the threshold indicates a higher probability of a sudden increase in battle fatalities. The results of this analysis demonstrate the ability of the LSTM Autoencoder model to discriminate "normal" days from "pre-unrest" days effectively. This is further supported by the ROC plot, which shows an **AUC very close to 1, at 0.9377**.

> **[Fig. 7]** Reconstruction error and ROC plot, Ukraine. **AUC: 0.9377.**
>
> **[그림 내용 — 좌측: 재구성 오차 플롯]** X축: 2021-07 ~ 2022-03, Y축: SE 0~25. 주황 원(normal days)이 대부분 SE 0~5 범위에 밀집. 2022년 1월부터 산호색 십자(pre-unrest days)가 나타나며 SE 5~20으로 상승. 파란 이동평균선(15일)이 2022-02 초부터 검은 threshold 선을 돌파하며 급상승 → 침공일(빨간 점선) 직전 최대값. 핵심: 침공 약 3~4주 전부터 이동평균이 threshold를 넘기 시작.
>
> **[그림 내용 — 우측: ROC 곡선]** X축: FPR(0~1), Y축: TPR(0~1). 곡선이 좌상단 모서리에 매우 근접 → AUC 0.9377. 대각선(random classifier) 대비 훨씬 우수한 분류 성능.

In the second case study, we aimed to investigate whether it was possible to predict the high number of battle fatalities prior to the Burkina Faso coup that occurred on January 23, 2022. The optimal values for the parameters Battle Fatalities Threshold and Pre-unrest days were once again determined through a Grid Search optimization technique in this case study. The case study evaluated the model's ability to detect the 63 "pre-unrest" days preceding the coup on January 23, 2022. The model was considered successful if it could classify a day as "pre-unrest" even if the number of battle fatalities on that specific day was < 52, as long as it was one of the 63 days leading up to an "unrest" day with > 52 battle fatalities (Table 4).

**Table 4.** Burkina Faso parameters.

| Battle fatalities threshold | Pre-unrest days |
|---|---|
| 52 (99.8 %)^a | 63 |

> ^a 99.8% of days in the entire dataset have battle fatalities < 52.

As shown in Fig. 8, the LSTM Autoencoder model effectively discriminates the "pre-unrest" days from the "normal" days, as highlighted by the reconstruction error plot. The 15-day moving average of the reconstruction error (the technique adopted in Fig. 7) surpasses the threshold in April 2021 and remains above the threshold throughout the period in question. This indicates a higher probability of approaching a situation of socio-political instability. Additionally, the model's performance can be seen from the ROC plot, with an **AUC of 0.8723**. It is important to note that while the trend of increasing battle fatalities in Burkina Faso may seem apparent, this case study tested the model's ability to predict such trends rather than sudden breaks in behavior patterns, as seen in the previous case study. Nevertheless, the results can be regarded as excellent given the high AUC value close to the maximum value of 1.

> **[Fig. 8]** Reconstruction error and ROC plot, Burkina Faso. **AUC: 0.8723.**
>
> **[그림 내용 — 좌측: 재구성 오차 플롯]** X축: 2020-11 ~ 2022-01, Y축: SE 0~80. 주황 원(normal)은 SE 0~15에 분포. 산호색 십자(pre-unrest)가 SE 10~70 범위로 높게 분포. 파란 이동평균선이 **2021-04부터 이미 threshold를 돌파**하고 이후 계속 threshold 위에 머무름. 빨간 점선(2021-08-18): 역대 최대 fatalities 147명. 빨간 점선(2022-01-19): 쿠데타 4일 전 107명. 핵심: 이 케이스는 갑작스러운 변화보다 장기 상승 추세를 감지한 사례.
>
> **[그림 내용 — 우측: ROC 곡선]** AUC 0.8723. 곡선이 좌상단에 근접하나 Ukraine보다는 약간 아래 — 장기 추세 탐지라 분리가 다소 덜 명확.

In the final case study, we aimed to predict the onset of violence in Sri Lanka that occurred on May 9, 2022, two days before the government's announcement that it was deploying the army with orders to shoot on sight to bring the violence under control. Once again in this final case study, the optimal values for the parameters Violence Threshold and Pre-Unrest Days were determined through a Grid Search optimization technique. The case study evaluated the model's ability to detect the 47 "pre-unrest" days preceding the onset of violence on May 9, 2022. The model was considered successful if it could classify a day as "pre-unrest" even if the amount of violence on that specific day was < 15, as long as it was one of the 47 days leading up to an "unrest" day with > 15 violent events (Table 5).

**Table 5.** Sri Lanka parameters.

| Violence threshold | Pre-unrest days |
|---|---|
| 15 (99.8 %)^a | 47 |

> ^a 99.8% of days in the entire dataset have violence < 15.

As can be observed from the plots in Fig. 9, the model performed exceptionally well. The reconstruction error plot clearly distinguishes between "normal" and "pre-unrest" days, and the ROC plot demonstrates an **AUC of 0.9191**. Additionally, this case study is evidence of the model's ability effectively to identify a breaking trend using the variable violence, which was derived from other variables in the ACLED dataset.

> **[Fig. 9]** Reconstruction error and ROC plot, Sri Lanka violence. **AUC: 0.8656.**
>
> **[그림 내용 — 좌측: 재구성 오차 플롯]** X축: 2022-01 ~ 2022-05, Y축: SE 0~14. 주황 원(normal)은 SE 0~4에 밀집. 산호색 십자(pre-unrest)가 SE 4~14 범위로 명확히 분리. 빨간 점선(2022-04-01): 비상사태 선포. 빨간 점선(2022-05-11): 군 투입. 파란 이동평균이 4월 초부터 threshold를 넘어 상승. 핵심: violence라는 파생 변수를 target으로 사용해도 pre-unrest 탐지가 효과적.
>
> **[그림 내용 — 우측: ROC 곡선]** AUC 0.8656. 세 케이스 중 가장 낮지만 여전히 우수한 분류 성능.

---

## 8. Conclusion

This paper presents an original method for predicting sudden increases in battle fatalities and violence using machine learning models. It should be noted that most studies in this field tend to adopt supervised approaches to conflict prediction, which require labeled data to learn patterns and make predictions. However, it turns out that this approach can be limited in detecting anomalous patterns, especially in contexts where anomalies are rare or ill-defined. We have therefore introduced the unsupervised method of anomaly detection for the first time in the study of conflict early warning. This allows us to explore the underlying structures in the data by analyzing the concept of a conflict anomaly, an unlabelled data, providing an additional tool to identify and address anomalies that may not be captured by traditional supervised approaches.

The anomaly detection approach uses historical data to identify a threshold for the target variable and then applies this threshold to predict the likelihood of an imminent dramatic increase. The results demonstrate that the models developed are highly accurate, with **AUC scores ranging from 87.2 % to 93.7 %** for predicting battle fatalities and **86.6 %** for predicting violence.

The key advantage of this approach is that it prioritizes the identification of sudden outbreaks, rather than predicting the exact number of occurrences. This is crucial for governments and international organizations, as it allows for early intervention and the implementation of preventative measures to mitigate the impact of an unrest event. This method should be used to complement, rather than replace, diplomatic efforts, by providing a quantifiable perspective on potential sociopolitical instability.

There are several avenues for future research that could further improve the accuracy of these models. For instance, creating new independent variables to consider the diplomatic relationships between neighboring countries, testing new variables derived from existing datasets, such as the violence variable, the integration with data that is more strictly economic, climatic or otherwise not geopolitical from a human perspective, or incorporating ensemble models that use multiple models instead of just the best-performing one.

Another possible future development in addition to extending it to more countries is to train the model by merging countries in such a way as to predict anomalies occurring in a similar way in other countries, the basic idea behind this possible development being that a crisis occurring for example in Nigeria could be similar to a crisis about to occur in Kenya.

Moreover, the outputs generated by these models can be used as independent variables in forecasting models aimed at predicting the exact number of occurrences. The squared error score assigned to each day can be used as an indicator of potential outbreaks, with a higher score identifying a higher likelihood of occurrence. Incorporating this unsupervised approach as input into classical supervised machine learning models has the potential to enhance the accuracy in predicting the exact number of occurrences (Boquet et al., 2020). In conclusion, the predictive diplomacy method presented in this paper has the potential to assist governments and international organizations in anticipating and preventing sociopolitical instability. Additionally, this method can also be applied in fields other than peacekeeping, such as in the procurement of rare earths, raw materials, and energy sources. For example, if a supplier country is identified as being susceptible to an outbreak, the importing country may be able to pursue the diversification of its sources proactively.

The development and implementation of these models should receive political attention and ought to be integrated into diplomatic efforts to ensure the safety and stability of nations and their citizens worldwide.

### Limitations

It is important to highlight that the conclusions drawn from this study are subject to some limitations. Firstly, we cannot assume that the characteristics of anomalous 60-day periods will repeat exactly at 60-day intervals in subsequent crises, nor can we guarantee that anomalous days will inevitably lead to a crisis. Further studies are needed to better understand the nature of the signal preceding a crisis and to more fully validate the proposed method. Additionally, it is important to note that the presence of an anomalous day does not necessarily guarantee the proximity of a crisis, as days, weeks, or even months may pass before a critical event occurs. Only through rigorous and comprehensive research can we improve our understanding and prediction capabilities of critical events. It is necessary to emphasize that our methodology may not be applicable in all situations or geopolitical contexts, and that customizations and adaptations may be required to make it effective in certain scenarios. Ultimately, while the preliminary results are very promising, it is essential to proceed with caution and continue to develop and enhance these models to ensure their reliability and utility in preventing international crises.

It is also for these reasons that a rigorous evaluation is planned in the coming months to assess this method's efficiency through live forecasting, rather than mere case studies. The ability to support diplomatic action in anticipating and preventing conflicts through data-driven decision-making has the potential to impact greatly the global security, stability, and resilience of all countries.

---

## CRediT authorship contribution statement

- **Luca Macis:** Data curation, Formal analysis, Methodology, Validation, Visualization, Writing – original draft, Writing – review & editing.
- **Marco Tagliapietra:** Data curation, Formal analysis, Methodology, Validation, Visualization, Writing – original draft, Writing – review & editing.
- **Rosa Meo:** Data curation, Formal analysis, Methodology, Supervision, Writing – original draft.
- **Paola Pisano:** Investigation, Supervision, Writing – original draft, Writing – review & editing.

---

## Acknowledgments

This study was funded by the European Union - NextGenerationEU, in the framework of the GRINS - Growing Resilient, INclusive and Sustainable project (GRINS PE00000018 - CUP D13C22002160001). The views and opinions expressed are solely those of the authors and do not necessarily reflect those of the European Union, nor can the European Union be held responsible for them. We would also like to thank the Crisis Unit of the Italian Ministry of Foreign Affairs and International Cooperation for their support in launching the predictive diplomacy project in 2022.

---

## References

- Ackermann, Alice, 2003. The idea and practice of conflict prevention. *J. Peace Res.* 40(3), 339–347.
- ACLED, 2022a. ACLED Coverage to Date.
- ACLED, 2022b. Data Export Tool.
- ACLED, 2022c. ACLED Trendfinder.
- Beck, Nathaniel, King, Gary, Zeng, Langche, 2000. Improving quantitative studies of international conflict: a conjecture. *Am. Polit. Sci. Rev.* 94 (March), 21.
- Boquet, Guillem, Morell, Antoni, Serrano, Javier, Vicario, Jose Lopez, 2020. A variational autoencoder solution for road traffic forecasting systems: missing data imputation, dimension reduction, model selection and anomaly detection. *Transp. Res. Part C Emerg. Technol.* 115, 102622.
- Compton, Ryan, Lee, Craig, Xu, Jiejun, Luis, Artieda Moncada, Lu, Tsai-Ching, 2014. Using Publicly Visible Social Media to Build Detailed Forecasts of Civil Unrest.
- GDELT, 2022. Documentation.
- George, Alexander L., 1999. Strategies for preventive diplomacy and conflict resolution: scholarship for policy-making. *Coop. Confl.* 34(1), 9–19.
- Gleditsch, Kristian Skrede, 2002. Expanded trade and GDP data. *J. Confl. Resolut.* 46(5), 712–724.
- Halkia, Matina, Ferri, Stefano, Schellens, Marie K., Papazoglou, Michail, Thomakos, Dimitrios, 2020a. The Global Conflict Risk Index: a quantitative tool for policy support on conflict prevention. *Prog. Disaster Sci.* 6 (April), 100069.
- Halkia, Matina, Ferri, Stefano, Schellens, Marie K., Papazoglou, Michail, Thomakos, Dimitrios, 2020b. The Global Conflict Risk Index. *Prog. Disaster Sci.* 6, 100069.
- Hegre, Håvard, Karlsen, Joakim, Nygård, Håvard Mokleiv, Strand, Håvard, Urdal, Henrik, 2012. Predicting armed conflict, 2010–2050. *Int. Stud. Q.* 57 (December), 250–270.
- Hegre, Håvard, et al., 2019a. ViEWS: a political violence early-warning system. *J. Peace Res.* 56(2), 155–174.
- Hegre, Håvard, Hultman, Lisa, Nygård, Håvard Mokleiv, 2019b. Evaluating the conflict-reducing effect of UN peacekeeping operations. *J. Polit.* 81(1), 215–232.
- Junior, Manoel, et al., 2021. Towards understanding the use of telegram by political groups in Brazil. *WebMedia '21*, 237–244.
- Khaund, Tuja, et al., 2021. Telegram: data collection, opportunities and challenges. *Information Management and Big Data*, pp. 513–526.
- Korkmaz, Gizem, et al., 2015. Combining heterogeneous data sources for civil unrest forecasting. *ASONAM '15*, 258–265.
- Leetaru, Kalev, Schrodt, Philip A., 2013. Gdelt: global data on events, location, and tone, 1979–2012. *ISA Annual Convention*, vol. 2.
- Matina, Halkia, et al., 2019. Dynamic Global Conflict Risk Index. Report. Joint Research Centre (JRC).
- Muchlinski, David, et al., 2016. Comparing random forest with logistic regression for predicting class-imbalanced civil war onset data. *Polit. Anal.* 24, 87–103.
- Nakazawa, Takeshi, Kulkarni, Deepak V., 2019. Anomaly detection and segmentation for wafer defect patterns using deep convolutional encoder–decoder neural network architectures in semiconductor manufacturing. *IEEE Trans. Semicond. Manuf.* 32(2), 250–256.
- Perry, Chris, 2013. Machine learning and conflict prediction: a use case. *Stability: Int. J. Secur. Dev.* 2 (October), 56.
- Raleigh, Kishi, Linke, 2023. Political instability patterns are obscured by conflict dataset scope conditions, sources, and coding choices. *Humanit. Soc. Sci. Commun.* 10, 74.
- Ranjan, Chitta, 2020. *Understanding deep learning: application in rare event prediction*, 1st. Connaissance Publishing.
- Rustad, Siri Camilla, et al., 2011. All conflict is local: modeling sub-national variation in civil conflict risk. *Confl. Manag. Peace Sci.* 28(1), 15–40.
- Sapiezynska, Ewa, Lagos, Claudia, 2016. Media freedom indexes in democracies: a critical perspective through the cases of Poland and Chile. *Int. J. Commun.* 10 (January), 22.
- Schneider, Laura, 2019. *Measuring Global Media Freedom: The Media Freedom Analyzer as a New Assessment Tool.* Book, pp. 1–252.
- Schrodt, Philip A., Analytics, Parus, 2015. Comparison Metrics for Large Scale Political Event Data Sets.
- Siegel, Barry, 2020. Industrial anomaly detection: a comparison of unsupervised neural network architectures. *IEEE Sens. Lett.* 4(8), 1–4.
- Wallensteen, Peter, et al., 2018. Violence, Wars, Peace, Security, vol. 2. Book chapter, pp. 411–456.
- Zhang, Ang, Zhao, Xiaoyong, Wang, Lei, 2021. CNN and LSTM based encoder-decoder for anomaly detection in multivariate time series. *ITNEC*, vol. 5, pp. 571–575.
