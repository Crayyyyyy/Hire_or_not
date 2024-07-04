import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load your data
data = pd.read_csv('./recruitment_data.csv')

# Histograms for all features
data.hist(figsize=(50,50))
plt.show()

# Scatter plot for a pair of variables, e.g., 'ExperienceYears' vs 'InterviewScore'
plt.figure(figsize=(4, 6))
sns.scatterplot(x='ExperienceYears', y='InterviewScore', data=data, hue='RecruitmentStrategy', palette="bright")
plt.title('Experience vs Score')
plt.show()
