## Hypotheses to Test

### H1: Sentiment Analysis Hypothesis

**Null Hypothesis:** There is no difference in average sentiment scores between performance-related repository commit messages and simplicity-related repository commit messages.

**Alternative Hypothesis:** Performance-related repositories have lower sentiment scores than simplicity-related repositories.

**Reasoning:** Performance-related repositories may be more technical and challenging to develop, potentially leading to more negative sentiment in commit messages, but we will test this with hypothesis testing.

---

### H2: Contributor Concentration Hypothesis

**Null Hypothesis:** There is no difference in contributor concentration (calculated by Gini coefficient) between performance-related and simplicity-related repositories.

**Alternative Hypothesis:** Performance-related repositories have a higher contributor concentration than simplicity-related repositories.

**Reasoning:** Performance-related repositories may require a small team of expert developers while simplicity-related repositories may have more diverse contributors as these packages can often be used more casually, but we will test this with hypothesis testing.

---

### H3: Weekend Activity Hypothesis

**Null Hypothesis:** There is no difference in the proportion of weekend activity (weekend activity/total activity) between performance-related and simplicity-related repositories.

**Alternative Hypothesis:** Simplicity-related repositories have a higher proportion of weekend activity than performance-related repositories.

**Reasoning:** Contributors may work on simplicity-related repositories more on weekends than performance-related repositories, as much of this work may be lighter and less taxing.

---

### H4: Time Series Trend Hypothesis

**Null Hypothesis:** There is no difference in the formalization ratio (proportion of activity conducted through pull requests) over time between performance-related and simplicity-related repositories.

**Alternative Hypothesis:** The formalization ratio increases over time for performance-related and decreases for simplicity-related repositories.

**Reasoning:** As machine learning has become more widespread in recent years, it is possible that the formal review (PRs) has increased for performance-related repositories (such as scikit-learn) compared to direct commits,  while it may have decreased for simplicity-related repositories (such as visualization tools), but we will test this with time series analysis and hypothesis testing.