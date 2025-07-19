# Copilot Project Context and Description

## Project Description
This project focuses on user behavior analysis on Instagram, with an emphasis on understanding and predicting user engagement. The workflow includes:
- Collecting and processing Instagram post data from user folders
- Merging post data with influencer metadata (username, category, followers, etc.)
- Cleaning and deduplicating the merged data
- Performing exploratory data analysis and visualizations to understand patterns in likes, comments, hashtags, sentiment, and user interactions
- Generating insights and suggestions for maximizing post popularity and engagement
- Applying clustering and machine learning models for user segmentation and engagement prediction

The main goal is to analyze how factors such as hashtags, posting time, user category, and sentiment influence engagement metrics (likes, comments, etc.) and to provide actionable recommendations for content optimization.

## Project Context
- Workspace root: /home/krishantha/Github/fyp-l4
- OS: Linux
- Default shell: bash
- Data is stored in the `data/` directory, with subfolders for clustered and processed data
- Analysis scripts are in the `analysis/` folder
- Data processing scripts are in the `scripts/` folder
- Visualizations are generated in the `visualizations/` and `outputs/` folders
- Jupyter notebooks for exploration are in the `notebooks/` folder
- Recommendations logic is in the `recommendations/` folder

This project supports a final year research module on user behavior analysis, engagement metrics, sentiment analysis, user segmentation, and AI-driven recommendations for Instagram.

---

## 2.4 Module 04: User Behavior Analysis

### 2.4.1 Metrics for Measuring User Engagement
Likes, comments, shares, and views are the measurement scales of user engagement on Instagram. Since Instagram doesn’t have a reaction-based system like Facebook and likes have been shown to imply positive responses such as appreciation or liking, comments will record a broad range of judgments ranging from negative to positive. The emphasis is made in analysis on how each of these measures needs to be analyzed on its own basis due to their dissimilar implications for user sentiment and conduct. Thus, for example, likes are apt to be correlated with passive use, whereas comments are more indicative of higher-order cognitive or affective engagement [5]. Furthermore, more advanced indicators of engagement - i.e., follower-adjusted interaction rates deliver a more differentiated picture of content performance by audience size [5].

### 2.4.2 Temporal Dynamics of Instagram Engagement
Temporal dynamics refers to how engagement with content changes over time. Research on Instagram's temporal engagement patterns illustrates that there are significant differences in the way users consume various types of content, such as photos, videos, and albums. Comparing these formats revealed that albums engage users for longer than photos and videos, whose spikes in engagement are fleeting. In addition, short content characterized by shorter captions and fewer hashtags consistently garners more engagement across all formats. These findings stress the necessity to tailor content approaches to specific media and user preferences in order to optimize immediate and sustained interactions [3].

### 2.4.3 Sentiment Analysis in Social Media Contexts
Sentiment analysis, the automated process of identifying the emotional tone expressed in text or other communication, is significant in determining how audiences emotionally feel about content on Instagram. Classifying responses and comments into positive, negative, or neutral sentiments aids in identifying audience appeal trends and emotional cues. More advanced techniques like multimodal sentiment analysis, which implies a combination of textual and visual factors, have been shown to enhance sentiment classification accuracy. For instance, combining sentiment ratings with user engagement metrics provides more insight into how emotional responses affect interactions such as likes and shares [26].
Sentiment analysis can also help to identify temporal patterns. Using time-series sentiment dynamics, anyone can detect shifts in audience perceptions or preferences to change the existing approach to content creation. This iterative process helps make content relevant and appealing to the audience, which keeps on changing its expectations [26].

### 2.4.4 User Segmentation for Personalized Content Strategies
User segmentation is the process based on shared characteristics, dividing a broad audience into smaller, similar groups. This is crucial to creating tailored experiences on Instagram. Clustering algorithms are most commonly used in segmenting users based on shared characteristics such as engagement behavior, sentiment in comments, and content interests. For example, segmentation of followers into distinct segments allows brands to tailor messaging based on the needs of a particular audience, and in turn, spark engagement [15].
Studies have established that segmentation not only improves targeting precision but also creates more substantial relationships between brands and audiences. For instance, fashion brands that have employed segmentation have managed to target high-value segments of customers and develop targeted campaigns that more meaningfully engage with these segments [15]. This use of segmentation has immediate applicability to the proposed research's focus on personalized post recommendations.

### 2.4.5 Factors Influencing Engagement
A number of variables affect user interaction on Instagram, such as post type, timing, message interactivity, and emotional content appeal. Interactivity, such as calls-to-action or provocative captions, has been found to have the greatest effect on engagement rates, followed by post timing and topic appeal. Post format (e.g., photo versus video) also affects but is relatively less significant [15].
These findings suggest that enhancing these factors has a strong capacity to enhance engagement results. For instance, user-friendly posts through questions or polls produce more levels of interaction than static posts [15].

### 2.4.6 AI-Driven Machine Learning approaches for Engagement Prediction and Recommendation Systems
Machine learning algorithms have been used to a great extent to predict levels of social media engagement on posts. Such algorithms typically involve attributes such as post content (hashtags, captions), timing, sentiment scores, and segmentation information for the users. Regression models and deep learning models have proven particularly effective at predicting likes or comments [15].
The addition of new attributes such as visual beauty or content complexity has also improved prediction accuracy. For example, studies have shown that visually engaging posts with balanced compositions attract more attention and enjoy greater engagement rates [15]. Such findings are crucial in the development of effective predictive models for Instagram.
In addition, Artificial Intelligence (AI) now is at the center of Instagram's content recommendation platforms that enable personalized user experiences. Machine learning (ML) algorithms analyze user actions such as likes, comments, shares, and watch time on posts to develop deep user profiles. ML algorithms power Instagram's Feed and Explore, where content is ranked based on user interests. For instance, Vecchi and Francisco-Maffezzolli (2020) demonstrated how interpretable ML models can be employed to categorize Instagram posts and forecast engagement metrics by learning features like hashtags, image composition, and user interaction history [9].
Personalization is also enhanced with AI-driven dynamic content rank algorithms. These recommend and rank posts that are personalized according to individual interests based on collaborative filtering and content-based filtering. By incorporating AI into Instagram's platform, it ensures that the recommendations adapt based on user behavior over time [18].

### 2.4.7 Predicting Engagement Using Machine Learning
Instagram engagement prediction has been of high interest for research since it impacts directly on maximizing content strategies. Purba et al. (2024) contrasted various regression techniques to forecast post Engagement Rate (ER) based on a global dataset. Hashtags, image quality, and past user were the features employed within the models and demonstrated that strong regression techniques could well-predict engagement trends [13].
Similarly, another study utilized linear regression models trained on features such as username, followers, posting rate, and posting time to make predictions regarding likes on posts. The model achieved high accuracy with Root Mean Square Error (RMSE) as the metric of performance [23]. These findings highlight the promise that ML techniques have in providing actionable insights for creators.
Besides regression models, CNNs have also been applied in the study of visual content of posts. For example, aesthetically appealing posts with symmetrical composition were found to elicit improved engagement. This points to the importance of incorporating visual data in ML models to generate more precise predictions [13]

### 2.4.8 Sentiment Analysis for Emotional Insights
Sentiment analysis is a critical component of Instagram audience response analysis. Built on Natural Language Processing (NLP), AI algorithms scan text data from captions and comments and classify sentiment as positive, negative, or neutral. More advanced sentiment analysis techniques also detect specific emotions like joy or anger by picking up on linguistic context.
A study by Purba et al. (2024) highlighted the role of sentiment analysis in predicting engagement trends. By combining sentiment scores with other features like hashtags and user demographics, the researchers improved the accuracy of their prediction models [9]. Multimodal sentiment analysis—integrating textual data with visual cues—has also been shown to provide deeper insights into how emotional responses influence engagement metrics [9].

### 2.4.9 User Segmentation Using Clustering Algorithms
User segmentation is important to tailor content strategies for different sets of audiences. Clustering methods such as K-Means or hierarchical clustering are typically used to segment users into groups according to shared characteristics such as engagement behavior and comment sentiment.
For example:
A study modeled Instagram’s influencer ecosystem as a graph using Node2Vec and Word2Vec algorithms to detect communities within the network. These clusters revealed patterns in influencer interactions and provided actionable insights for optimizing marketing strategies [27].
Behavioral segmentation identified high-value followers who consistently engage with posts, enabling brands to target these segments with personalized campaigns [27].
Segmentation not only improves targeting precision but also enhances recommendation systems by aligning them with specific audience clusters.

### 2.4.10 Recommendation Systems for Post Suggestions
AI-powered recommendation systems analyze historical user data to suggest optimal post characteristics such as captions, hashtags, media types, and posting times. For instance:
Vecchi et al. (2020) developed interpretable ML algorithms that classified Instagram posts based on their likelihood of generating engagement [9].

Another study utilized collaborative filtering techniques to recommend hashtags aligned with trending topics or user interests[18].
These systems adapt dynamically by incorporating real-time feedback from user interactions, ensuring that recommendations remain relevant over time.

### 2.4.11 Graph-Based Approaches for Influencer Networks
Graph theory combined with ML has been applied to model Instagram’s influencer ecosystem. A study by IIETA (2024) used graph-based approaches like Node2Vec to analyze influencer networks and predict collaboration opportunities within communities [27]. The research revealed key structural characteristics of influencer networks—such as centrality measures—that influence engagement dynamics.
By leveraging graph analytics, brands can identify emerging influencers and optimize their collaboration strategies within specific niches.

### 2.4.12 Personalized Recommendations for Content Optimization
Personalized recommendation systems utilize insights derived from sentiment analysis and user segmentation to recommend post characteristics that ought to be optimal for style, captions, hashtags, and media types. Research indicates that personalized recommendation boosts user engagement by ensuring content alignment with individual preferences [26].
For instance, systems that analyze past user interactions can predict which types of posts are likely to resonate with specific audience segments. This approach not only improves engagement metrics but also fosters long-term loyalty by delivering consistently relevant experiences [26].

### 2.5 Summary
Throughout this chapter, we have discussed the research completed under each module domain. Additionally, we have included how our research will contribute to research progress by adding novelty.
